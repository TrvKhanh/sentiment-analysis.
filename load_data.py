import pandas as pd
import re
import json
from datasets import Dataset
import os
import py_vncorenlp



VNCORE_DIR = "/home/big/pytorch/fine-tune/vncorenlp"

os.makedirs(VNCORE_DIR, exist_ok=True)

if not os.path.exists(os.path.join(VNCORE_DIR, "VnCoreNLP-1.2.jar")):
    print("Downloading VnCoreNLP model...")
    py_vncorenlp.download_model(save_dir=VNCORE_DIR)
else:
    print("VnCoreNLP model already exists.")

rdrsegmenter = py_vncorenlp.VnCoreNLP(
    annotators=["wseg"],
    save_dir=VNCORE_DIR
)


def clear_text(text: str, emoji_map: dict, emoji_pattern: re.Pattern) -> str:
    text = emoji_pattern.sub(lambda m: f" {emoji_map[m.group(0)]} ", text)

    text = re.sub(r"[^a-zA-Z0-9À-ỹ_\s\.,!?()]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    sentences = rdrsegmenter.word_segment(text)
    text = " ".join(sentences)
    
    return text.lower()

def processing(csv_path: str, emoji_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.replace(r'^\s*$', pd.NA, regex=True).dropna().reset_index(drop=True)

    with open(emoji_path, "r", encoding="utf-8") as f:
        emoji_map = json.load(f)
    emoji_pattern = re.compile("|".join(map(re.escape, emoji_map.keys())))

    df["Review"] = df["Review"].map(lambda x: clear_text(x, emoji_map, emoji_pattern))
    return df

def prepare_dataset(csv_path: str, emoji_path: str) -> Dataset:
    df_clean = processing(csv_path, emoji_path)

    df_clean["labels_aspect"] = df_clean.apply(lambda x: [
        1 if x["giai_tri"] > 0 else 0,
        1 if x["luu_tru"] > 0 else 0,
        1 if x["nha_hang"] > 0 else 0,
        1 if x["an_uong"] > 0 else 0,
        1 if x["van_chuyen"] > 0 else 0,
        1 if x["mua_sam"] > 0 else 0,
    ], axis=1)

    df_clean["labels_sentiment"] = df_clean.apply(lambda x: [
        x["giai_tri"], x["luu_tru"], x["nha_hang"],
        x["an_uong"], x["van_chuyen"], x["mua_sam"]
    ], axis=1)

    return Dataset.from_pandas(df_clean)
