import os
import torch
import torch.nn as nn
import numpy as np
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from huggingface_hub import login
from dotenv import load_dotenv
from datasets import DatasetDict
from load_data import prepare_dataset
import wandb
import logging
from datetime import datetime
from sklearn.metrics import f1_score


logger = logging.getLogger("TrainLogger")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
os.makedirs("logs", exist_ok=True)
file_handler = logging.FileHandler("logs/train.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)



load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
    logger.info("Logged into HuggingFace Hub successfully")
else:
    logger.warning("HF_TOKEN not found in .env!")

wandb.login()
wandb.init(
    project="phobert-aspect-sentiment",
    name="PhoBERT_ABSA_Uncertainty",
    notes="PhoBERT ABSA with learnable uncertainty weighting",
    tags=["phobert", "absa", "uncertainty-weighting"],
)
logger.info(f"WandB run initialized: {wandb.run.name}")

# -----------------------
# Device
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.warning("CUDA not available, using CPU")




MODEL_NAME = "vinai/phobert-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)

for param in base_model.parameters():
    param.requires_grad = False


class AnalystModel(nn.Module):
    def __init__(self, pretrain, num_aspects=6, pos_weight=None):
        super().__init__()
        self.model = pretrain
        self.hidden_size = self.model.config.hidden_size
        self.num_aspects = num_aspects

        
        self.aspect_head = nn.Linear(self.hidden_size, num_aspects)
        self.sentiment_head = nn.Linear(self.hidden_size, num_aspects)

       
        self.log_var_cls = nn.Parameter(torch.zeros(1))
        self.log_var_sent = nn.Parameter(torch.zeros(1))

        
        if pos_weight is not None:
            self.loss_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            self.loss_cls = nn.BCEWithLogitsLoss()
        self.loss_sent = nn.MSELoss(reduction="none")

    def forward(self, input_ids, attention_mask,
                aspect_labels=None, sentiment_labels=None, **kwargs):
        
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls_emb = outputs.last_hidden_state[:, 0, :]  # [batch, hidden]

        aspect_logits = self.aspect_head(cls_emb)      # [batch, num_aspects]
        sentiment_preds = self.sentiment_head(cls_emb) # [batch, num_aspects] 
        if aspect_labels is None and sentiment_labels is None:
            return (aspect_logits, sentiment_preds)

        
        mask = aspect_labels.float().to(aspect_logits.device)  # (batch, num_aspects)

        # classification loss
        L_cls = self.loss_cls(aspect_logits, mask)

        
        sent_raw = sentiment_labels.float().to(aspect_logits.device)
        sent_norm = torch.zeros_like(sent_raw)
        nonzero_mask = (sent_raw > 0)
        sent_norm[nonzero_mask] = (sent_raw[nonzero_mask] - 1.0) / 4.0

        sent_prob = torch.sigmoid(sentiment_preds)
        mse = self.loss_sent(sent_prob, sent_norm)  # (batch, num_aspects)
        masked_mse = (mse * mask).sum() / mask.sum().clamp(min=1.0)

        
        loss_total = (torch.exp(-self.log_var_cls) * L_cls + self.log_var_cls) + \
                     (torch.exp(-self.log_var_sent) * masked_mse + self.log_var_sent)

        return {
            "loss": loss_total,
            "loss_aspect": L_cls.detach() if isinstance(L_cls, torch.Tensor) else L_cls,
            "loss_sent": masked_mse.detach() if isinstance(masked_mse, torch.Tensor) else masked_mse,
            "log_var_cls": self.log_var_cls,
            "log_var_sent": self.log_var_sent,
            "aspect_logits": aspect_logits,
            "sentiment_preds": sentiment_preds,
        }


dataset = prepare_dataset(
    csv_path="/home/big/pytorch/fine-tune/train-problem.csv",
    emoji_path="/home/big/pytorch/fine-tune/emoji.json"
)

def tokenizer_data(batch):
    encoding = tokenizer(
        batch["Review"],
        truncation=True,
        max_length=256,
        padding=False,
    )
    encoding["labels_aspect"] = batch["labels_aspect"]
    encoding["labels_sentiment"] = batch["labels_sentiment"]
    return encoding

dataset = dataset.map(tokenizer_data, batched=True)

_required_cols = {"input_ids", "attention_mask", "labels_aspect", "labels_sentiment"}
_cols_to_drop = [c for c in dataset.column_names if c not in _required_cols]
dataset = dataset.remove_columns(_cols_to_drop)

if not isinstance(dataset, DatasetDict):
    dataset = DatasetDict(dataset.train_test_split(test_size=0.1, seed=42))

train_dataset = dataset["train"]
eval_dataset = dataset["test"]

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)


class AnalystTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        labels_aspect = inputs.pop("labels_aspect").to(model.aspect_head.weight.device).float()
        labels_sentiment = inputs.pop("labels_sentiment").to(model.aspect_head.weight.device).float()

        outputs = model(**inputs, aspect_labels=labels_aspect, sentiment_labels=labels_sentiment)
        loss = outputs["loss"]

        
        logs = {}
        if outputs.get("loss_aspect") is not None:
            try:
                logs["train/loss_aspect"] = float(outputs["loss_aspect"].detach().cpu().item())
            except Exception:
                logs["train/loss_aspect"] = float(outputs["loss_aspect"])
        if outputs.get("loss_sent") is not None:
            try:
                logs["train/loss_sent"] = float(outputs["loss_sent"].detach().cpu().item())
            except Exception:
                logs["train/loss_sent"] = float(outputs["loss_sent"])
        
        logs["train/sigma_cls"] = float(torch.exp(outputs["log_var_cls"]).detach().cpu().item())
        logs["train/sigma_sent"] = float(torch.exp(outputs["log_var_sent"]).detach().cpu().item())
        if loss is not None:
            logs["train/loss_total"] = float(loss.detach().cpu().item())

        if len(logs) > 0:
            wandb.log(logs)

        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    """
    eval_pred: (preds, labels)
    preds may be:
      - tuple/list: (aspect_logits_np, sentiment_preds_np)
      - np.ndarray (rare) - not expected
      - dict-like (rare)
    labels may be:
      - dict with keys 'labels_aspect' and 'labels_sentiment'
      - tuple/list: (aspect_labels_np, sentiment_labels_np)
      - np.ndarray stacked (not expected)
    """
    preds, labels = eval_pred

    # ---------- normalize preds ----------
    if isinstance(preds, (list, tuple)) and len(preds) == 2:
        aspect_logits_np = np.asarray(preds[0])
        sentiment_preds_np = np.asarray(preds[1])
    elif isinstance(preds, dict):
        aspect_logits_np = np.asarray(preds.get("aspect_logits"))
        sentiment_preds_np = np.asarray(preds.get("sentiment_preds"))
    else:
        # fallback: try to convert
        preds_arr = np.asarray(preds)
        # try to split in half if second dim is 2*num_aspects
        if preds_arr.ndim == 3 and preds_arr.shape[1] == NUM_ASPECTS:
            # maybe already (B, num_aspects, 2) unlikely
            aspect_logits_np = preds_arr[:, :, 0]
            sentiment_preds_np = preds_arr[:, :, 1]
        else:
            raise ValueError("Unexpected preds format for compute_metrics. Got shape: {}".format(preds_arr.shape))

    # ---------- normalize labels ----------
    if isinstance(labels, dict):
        aspect_labels_np = np.asarray(labels["labels_aspect"])
        sentiment_labels_np = np.asarray(labels["labels_sentiment"])
    elif isinstance(labels, (list, tuple)) and len(labels) == 2:
        aspect_labels_np = np.asarray(labels[0])
        sentiment_labels_np = np.asarray(labels[1])
    else:
        # Sometimes Trainer packs labels as ndarray with shape (B, something)
        # We assume first NUM_ASPECTS cols are aspect, next NUM_ASPECTS are sentiment (if that matches)
        lbl_arr = np.asarray(labels)
        if lbl_arr.ndim == 2 and lbl_arr.shape[1] == 2 * NUM_ASPECTS:
            aspect_labels_np = lbl_arr[:, :NUM_ASPECTS]
            sentiment_labels_np = lbl_arr[:, NUM_ASPECTS:]
        else:
            raise ValueError("Unexpected labels format for compute_metrics. Got shape: {}".format(lbl_arr.shape))

    # ---------- Aspect Micro-F1 ----------
    # apply sigmoid then 0.5 threshold
    aspect_probs = 1 / (1 + np.exp(-aspect_logits_np))
    aspect_preds_bin = (aspect_probs > 0.5).astype(int)
    aspect_true = aspect_labels_np.astype(int)

    # handle edge case when there are no positive labels in ground-truth
    try:
        micro_f1 = f1_score(aspect_true.flatten(), aspect_preds_bin.flatten(), average="micro", zero_division=0)
    except Exception:
        micro_f1 = 0.0

    # ---------- Sentiment Quality ----------
    # map model preds (sigmoid 0..1) -> 1..5 and round
    sent_probs = 1 / (1 + np.exp(-sentiment_preds_np))
    sent_scaled = 1 + 4 * sent_probs  # in [1,5]
    sent_rounded = np.rint(sent_scaled).astype(int)
    sent_rounded = np.clip(sent_rounded, 1, 5)

    sent_true = sentiment_labels_np.astype(int)
    valid_mask = sent_true > 0  # only positions where GT > 0

    if valid_mask.sum() == 0:
        sentiment_acc = 0.0
    else:
        correct = (sent_rounded == sent_true) & valid_mask
        sentiment_acc = correct.sum() / valid_mask.sum()

        eval_overall = 0.7 * micro_f1 + 0.3 * sentiment_acc

    # log to wandb as well
    wandb.log({
        "eval/micro_f1": micro_f1,
        "eval/sentiment_acc": sentiment_acc,
        "eval/overall": eval_overall
    })

    return {
        "micro_f1": float(micro_f1),
        "sentiment_acc": float(sentiment_acc),
        "overall": float(eval_overall),
    }

# -----------------------
# Training arguments
# -----------------------
time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
training_args = TrainingArguments(
    output_dir=f"/home/big/pytorch/fine-tune/checkpoint/{time_str}",
    eval_strategy="epoch",             # ✅ đúng key (eval_strategy → evaluation_strategy)
    save_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir=f"/home/big/pytorch/fine-tune/logs{time_str}",
    logging_strategy="steps",
    logging_steps=100,
    warmup_ratio=0.03,
    fp16=True,
    report_to="wandb",
    remove_unused_columns=False,

    # ✅ Bổ sung 3 dòng quan trọng
    load_best_model_at_end=True,             # để early stopping & best model hoạt động
    metric_for_best_model="overall",         # dùng metric ta tính trong compute_metrics
    greater_is_better=True,                  # vì overall cao hơn là tốt hơn
)


# -----------------------
# Init model & Trainer
# -----------------------
model = AnalystModel(base_model, num_aspects=NUM_ASPECTS)

trainer = AnalystTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# -----------------------
# Run
# -----------------------
if __name__ == "__main__":
    trainable = [(name, p.shape) for name, p in model.named_parameters() if p.requires_grad]
    logger.info(f"Trainable parameters: {trainable}")

    trainer.train()
    wandb.finish()
    logger.info("✅ Training finished successfully!")
