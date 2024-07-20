"""
team-hatakeyama-phase2/Synthetic-TextWebImagesの準備
python tools/Synthetic-TextWebImages/to_llava_format.py

"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

INSTRUCTIONS = {
    "text": [
        "テキストを抽出してください。",
        "文字を認識して、テキストファイルに変換してください。",
        "OCR処理を実行してください。",
        "読み取ったテキストを表示してください。",
        "文字情報を取得してください。",
        "テキストを読み取ってください。",
        "OCRして、文字データを抽出してください。",
        "文字を認識し、テキストデータに変換してください。",
        "OCR解析を行い、テキストを取得してください。",
        "文字を検出し、テキスト化してください。",
        "文字情報を読み取ってください。",
        "テキストデータを生成してください。",
        "OCRを実行して、文字を取得してください。",
        "認識されたテキストを表示してください。",
        "文字認識を行ってください。",
        "文字を抽出して、テキストに変換してください。",
        "OCR技術を使ってテキストを取得してください。",
        "文字を読み取ってテキストデータにしてください。",
        "テキストを解析して取得してください。",
        "OCR処理で文字を認識してください。",
        "文字を読み取ってテキスト化してください。",
        "文字情報をテキストデータに変換してください。",
        "OCR解析でテキストを抽出してください。",
        "文字認識を行い、テキストファイルを作成してください。",
        "OCR技術で文字情報を抽出してください。",
        "文字を認識して、テキスト形式に変換してください。",
        "OCRを使用して文字データを取得してください。",
        "テキスト認識を実行してください。",
        "文字情報を読み取ってテキスト化してください。",
        "OCR処理を行い、テキストデータを生成してください。",
    ],
}

DATA_PATH = "./dataset/Synthetic-TextWebImages/ner-wikipedia-dataset"


def is_monochrome(image):
    pixels = image.getdata()

    first_pixel = pixels[0]

    for pixel in pixels:
        if pixel != first_pixel:
            return False
    return True


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

    dataset = load_dataset("team-hatakeyama-phase2/Synthetic-TextWebImages", "ner-wikipedia-dataset", cache_dir="./cache")

    llava_formats = {key: [] for key in INSTRUCTIONS.keys()}

    for i, data in tqdm(enumerate(dataset["train"])):
        if not is_monochrome(data["image"]):
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

                if len(answer) <= 2000:
                    llava_formats[key].append(
                        create_llava_format(random.choice(INSTRUCTIONS[key]), answer, image_filename)
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
