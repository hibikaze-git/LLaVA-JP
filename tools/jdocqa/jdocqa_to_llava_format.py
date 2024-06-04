"""
jdocqaの準備
python tools/jdocqa/jdocqa_to_llava_format.py --only_answerable
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset


def create_llava_format(data):
    llava_format = {}
    conversations = []

    conversation_user = {"from": "ユーザー", "value": f'{data["question"]}\n<image>'}

    conversation_system = {"from": "システム", "value": data["original_answer"]}
    conversations.append(conversation_user)
    conversations.append(conversation_system)

    pdf_path = Path(data["pdf_filepath"])

    llava_format["image"] = f"{pdf_path.stem + '_' + data['question_page_number']}.jpg"
    llava_format["conversations"] = conversations

    # print(llava_format)

    return llava_format


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate LLAVA format dataset")
    parser.add_argument('--with_ocr', action='store_true', help='Include OCR data in the output')
    parser.add_argument('--only_answerable', action='store_true', help='reject unanswerable data in the output')
    args = parser.parse_args()

    dataset = load_dataset(
        path="shunk031/JDocQA",
        # Rename to the same wording as in the paper: Document -> Report / Kouhou -> Pamphlet
        rename_pdf_category=True,
        # Set to True to use loading script for huggingface datasets
        trust_remote_code=True,
        cache_dir="./cache"
    )

    ocr_instructions = [
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

    llava_formats = []

    # VQA
    for data in dataset["train"]:
        do_append = True

        if args.only_answerable and data["no_reason"] == 0:
            do_append = False

        if do_append:
            llava_formats.append(create_llava_format(data))

    # OCR
    if args.with_ocr:
        for data in dataset["train"]:
            data["question"] = random.choice(ocr_instructions)

            if data["text_from_ocr_pdf"] != "" and data["text_from_ocr_pdf"] is not None:
                data["original_answer"] = data["text_from_ocr_pdf"]

                llava_formats.append(create_llava_format(data))

    print(len(llava_formats))

    output_filename = "jdocqa"

    if args.only_answerable:
        output_filename = output_filename + "_only_answerable"

    caption_path = Path("./dataset/jdocqa", output_filename + ".json")

    with open(caption_path, mode="w") as f:
        json.dump(llava_formats, f, indent=2, ensure_ascii=False)
