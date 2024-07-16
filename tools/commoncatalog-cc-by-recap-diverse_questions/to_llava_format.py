"""
commoncatalog-cc-by-recap-diverse_questionsの準備
python tools/commoncatalog-cc-by-recap-diverse_questions/to_llava_format.py

========== count qa pairs ==========
====================================
"""

import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

INSTRUCTIONS = {
    "question": None,
}

DATA_PATH = "./dataset/commoncatalog-cc-by-recap-diverse_questions"


def check_abnormal_answer(data):
    answer_ja = data
    #answer_en = data[key.replace("_ja", "")].lower()

    curation_list_ja = [
        #"ｔ個",
        #"知りません",
        #"ￒ",
        #"わかりません",
        #"分かりません",
        "ごめんなさい",
        "すみません",
        #"ｒ本",
        #"ｓ匹",
        #"ｒ匹",
        #"ｒつ",
        #"説明はつけない",
        "申し訳",
        "訳され",
    ]

    #curation_list_en = [
        #"translat",
    #]

    for item in curation_list_ja:
        if item in answer_ja:
            return True

    #for item in curation_list_en:
    #    if item in answer_en:
    #        return True

    return False


def remove_prefix(text, prefix):
    pattern = f'^{re.escape(prefix)}'
    return re.sub(pattern, '', text)


def remove_abnormal_sentence(text):
    # 「視覚的なAIアシスタント」という言葉が含まれるセンテンスを除去
    sentence_list = text.split("。")

    removed_sentence_list = []

    remove_sentence_phrase_list = [
        "何か他に知りたいことがあれば",
        "AIアシスタント",
    ]

    for sent in sentence_list:
        add_flag = True

        for remove_sentence_phrase in remove_sentence_phrase_list:
            if remove_sentence_phrase in sent.replace(" ", ""):
                add_flag = False

        if add_flag:
            removed_sentence_list.append(sent)

    removed_text = "。".join(removed_sentence_list)

    remove_phrase_list = [
        "もちろんです。",
        "もちろんです!",
        "もちろんです！",
        "了解しました。",
        "了解しました!",
        "了解しました！",
        "こんにちは。",
        "こんにちは!",
        "こんにちは！",
        "それでは、",
    ]

    for phrase in remove_phrase_list:
        removed_text = remove_prefix(removed_text, phrase)

    return removed_text.strip()


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

    dataset = load_dataset(
        "team-hatakeyama-phase2/commoncatalog-cc-by-recap-diverse_questions", cache_dir="./cache"
    )

    llava_formats = {key: [] for key in INSTRUCTIONS.keys()}

    for data in dataset["train"]:
        image_filename = str(data["photoid"]) + "." + "jpg"
        for key in INSTRUCTIONS.keys():
            answer = remove_abnormal_sentence(data["answer"])

            # 不要なデータを除去
            if check_abnormal_answer(answer):
                #print(answer)
                #print("===============")
                continue

            # 文章が長過ぎるデータは除去
            if len(answer) <= 2000:
                llava_formats[key].append(
                    create_llava_format(data[key], answer, image_filename)
                )

    print("========== count qa pairs ==========")
    for key, item in llava_formats.items():
        print(key, len(item))
    print("====================================")

    for key in INSTRUCTIONS.keys():
        output_filename = key

        caption_path = Path(
            DATA_PATH, output_filename + ".json"
        )

        with open(caption_path, mode="w") as f:
            json.dump(llava_formats[key], f, indent=2, ensure_ascii=False)
