import json

from datasets import load_dataset


dataset = load_dataset("toshi456/LLaVA-CC3M-Pretrain-595K-JA", cache_dir="./cache")

dataset_list = []

for i, data in enumerate(dataset["train"]):
    dataset_list.append(data)

with open("./dataset/LLaVA-CC3M-Pretrain-595K/chat_ja_calm2.json", "w", encoding="utf-8") as f:
    json.dump(dataset_list, f, indent=2, ensure_ascii=False)
