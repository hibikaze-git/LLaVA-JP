"""
anime-with-caption-cc0の準備
python tools/anime-with-caption-cc0/to_llava_format.py
"""

import argparse
import json
import os
import random
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

INSTRUCTIONS = {
    "phi3_caption_ja": "画像について詳細に説明してください。",
}

DATA_PATH = "./dataset/anime-with-caption-cc0"


def create_llava_format(question, answer, image_filename):
    """
    Args:
        question (str): 質問プロンプト
        answer (str): 回答
        image_filename (str): 拡張子付きの画像ファイル名(image.jpgなど)
    """
    llava_format = {}
    conversations = []

    conversation_user = {"from": "ユーザー", "value": f"{question}\n<image>"}

    conversation_system = {"from": "システム", "value": answer}
    conversations.append(conversation_user)
    conversations.append(conversation_system)

    llava_format["image"] = image_filename
    llava_format["conversations"] = conversations

    # print(llava_format)

    return llava_format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLAVA format dataset")
    args = parser.parse_args()

    dataset = load_dataset("alfredplpl/anime-with-caption-cc0", cache_dir="./cache")

    llava_formats = {key: [] for key in INSTRUCTIONS.keys()}

    for i, data in tqdm(enumerate(dataset["train"])):
        ext = data["image"].format

        if ext == "JPEG":
            image_filename = str(i) + "." + "jpg"
        elif ext == "PNG":
            image_filename = str(i) + "." + "png"
        else:
            print("unknown ext")
            raise

        image_output_path = os.path.join(DATA_PATH, "images", image_filename)

        data["image"].save(image_output_path)

        for key in INSTRUCTIONS.keys():
            answer = data[key]

            llava_formats[key].append(
                create_llava_format(INSTRUCTIONS[key], data[key], image_filename)
            )

    print("========== count qa pairs ==========")
    for key, item in llava_formats.items():
        print(key, len(item))
    print("====================================")

    for key in INSTRUCTIONS.keys():
        output_filename = key

        caption_path = Path(DATA_PATH, output_filename + ".json")

        with open(caption_path, mode="w") as f:
            json.dump(llava_formats[key], f, indent=2, ensure_ascii=False)
