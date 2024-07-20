"""
TextOCRの準備
python tools/TextOCR/to_llava_format.py

========== count qa pairs ==========
21584
====================================
"""

import argparse
import json
import random
from pathlib import Path


INSTRUCTIONS = [
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
]

DATA_PATH = "./dataset/TextOCR"


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

    with open("./dataset/TextOCR/TextOCR_0.1_train.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)

    filename_strings_dict = {}

    print(len(dataset["anns"]))

    for key, val in dataset["anns"].items():
        utf8_string = val["utf8_string"].strip()
        image_id = val["image_id"]
        file_name = dataset["imgs"][str(image_id)]["file_name"].replace("train/", "")

        if file_name not in filename_strings_dict:
            filename_strings_dict[file_name] = []

        if utf8_string.replace(" ", "") not in ["", "."]:
            filename_strings_dict[file_name].append(utf8_string)

    print(len(filename_strings_dict))

    llava_formats = []

    for image_filename, strings in filename_strings_dict.items():
        joined_strings = " ".join(strings)

        if len(joined_strings) <= 2000:
            answer = joined_strings

            llava_formats.append(
                create_llava_format(random.choice(INSTRUCTIONS), answer, image_filename)
            )

    print("========== count qa pairs ==========")
    print(len(llava_formats))
    print("====================================")

    caption_path = Path(DATA_PATH, "text_ocr_llava_format" + ".json")

    with open(caption_path, mode="w") as f:
        json.dump(llava_formats, f, indent=2, ensure_ascii=False)
