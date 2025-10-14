import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import login
from dotenv import load_dotenv
import os
from load_data import prepare_dataset

load_dotenv()
# Login Hugging Face
hf_token  = os.getenv("HF_TOKEN")
login(token=hf_token)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# Load pretrained PhoBERT
MODEL_NAME = "vinai/phobert-base"
NUM_ASPECTS = 6

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)

# Freeze all pretrained layers
for name, param in base_model.named_parameters():
    param.requires_grad = False


# Define custom AnalystModel
class AnalystModel(nn.Module):
    def __init__(self, pretrain):
        super(AnalystModel, self).__init__()
        self.model = pretrain
        self.hidden_size = self.model.config.hidden_size
        output = self.model
        self.aspects = nn.Sequential(
            nn.Linear(self.hidden_size, 6),
            nn.Sigmoid()
        )

        self.sentiment = nn.Sequential(
            nn.Linear()
        )
        

dataset = prepare_dataset(
    csv_path="/home/big/pytorch/fine-tune/train-problem.csv",
    emoji_path="/home/big/pytorch/fine-tune/emoji.json"
)

print(dataset[0])

