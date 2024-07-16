"""
team-hatakeyama-phase2/list_items_one_by_one_jaの準備
python tools/list_items_one_by_one_ja/to_llava_format_labels.py

========== count qa pairs ==========
labels_ja 30369
====================================
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

random.seed(42)

INSTRUCTION_POOLS = [
    "画像の各視覚的オブジェクトの中心に明るい数字のIDを付けました。それらの名前を列挙してください。",
    "画像の各アイテムには中心に個別の数字の識別子が付けられています。これらのタグに従ってアイテムの名前を順に列挙してください。",
    "写真の各視覚的オブジェクトに一意の数字のIDが中心に配置されています。割り当てられた番号でこれらのオブジェクトを列挙してください。",
    "画像内のすべてのオブジェクトには中心に数字のIDがタグ付けされています。これらの数字タグに基づいて各オブジェクトを特定し、リストアップしてください。",
    "画像内の各オブジェクトに中央に数字のIDがラベル付けされています。これらの数字IDを参照してすべてのオブジェクトをリストアップできますか？",
    "この画像に見えるすべてのオブジェクトには、中心に数字のIDが付けられています。対応する番号でオブジェクトをカタログしてください。",
    "この画像には、中心に数字のIDがハイライトされたさまざまなオブジェクトがあります。これらの数字識別子を使用してオブジェクトを項目化できますか？",
    "画像の各オブジェクトを特定し、各オブジェクトが中心に数字のIDでラベル付けされていることを確認してください。これらの数字タグに従ってオブジェクトをリストアップしてください。",
    "この画像の各オブジェクトの中心に数字のIDを付けました。これらの数字IDを言及してオブジェクトをリストアップできますか？",
    "この画像では、各オブジェクトの中心に数字のIDが付けられています。これらの数字IDに従ってオブジェクトを順にリストアップしてください。",
    "画像内の各オブジェクトに明るい数字のIDが中央に配置されています。これらのIDを参照してオブジェクトを列挙してください。",
    "この画像には、さまざまなオブジェクトと背景の部分に数字のタグが配置されています。これらのタグが表すもののリストを提供してください。",
    "提供された画像では、特定のオブジェクトや背景の要素の上に数字のタグが配置されています。これらのタグ付きエンティティを特定し、リストアップしてください。",
    "この画像では、一部のオブジェクトや背景の領域に数字のタグが見えます。タグ付けされたアイテムや要素の名前をリストアップしてください。",
    "画像内の複数のオブジェクトや背景の詳細に数字のタグが付けられています。それぞれを名前でリストアップできますか？",
    "異なるオブジェクトや背景を数値でタグ付けされた画像を見てください。これらのタグ付けされたものの名前を特定し、記録してください。",
    "画像には、さまざまなオブジェクトや背景の部分に数字のタグが付けられています。これらのタグに対応するオブジェクトやエリアを列挙してください。",
    "画像の特定のアイテムや背景部分に数字のタグが付いているのがわかりますか？これらのタグ付けされたオブジェクトやエリアの名前をリストアップしてください。",
    "この画像には、一部のオブジェクトや背景の特徴に数字のタグが付いています。これらのタグ付けされたエンティティの名前をカタログにしてください。",
    "この画像では、一部のアイテムや背景要素に数字のタグが割り当てられています。各タグに対応する名前をリストアップできますか？",
    "提供された画像の一部のオブジェクトや背景の詳細には数字のタグが付けられています。これらの数値タグが付いたアイテムや特徴の名前をリストアップしてください。",
    "画像内のオブジェクトに配置された数字の識別子に注目してください。これらの番号に対応する名前を詳述してください。",
    "この画像には、特定の特徴やオブジェクトに数字のIDが付けられています。各番号が何を指しているのかをアウトラインできますか？",
    "この写真には、アイテムや背景の特徴に数字のラベルが表示されています。これらの数字が示すエンティティをカタログ化できますか？",
    "数字のタグでマークされたアイテムやエリアがある画像が提供されています。これらのマークされた対象を特定し、列挙してください。",
    "この画像には、アイテムや背景の側面に番号が付けられています。これらの数字ラベルが何を意味するかをリストアップしてください。",
    "添付の画像にはさまざまなオブジェクトに数字のラベルが付けられています。これらのラベルが何を識別しているのかを解読し、リストアップしてください。",
    "数字のIDが見える画像を確認してください。それらは特定のアイテムや領域をマークしています。それぞれのタグが表すものをリストアップしてください。",
    "この画像には、特定のオブジェクトや背景要素に数字のタグが付けられています。これらのタグ付けされた場所やアイテムを概説してください。",
    "アイテムや背景に数字のタグが散らばっている画像を見てください。タグ付けされたアイテムのリストを作成できますか？",
    "画像内の各数字のIDは、オブジェクトまたはエリアに関連しています。これらのIDに基づいて特定し、リストを作成してください。",
    "提供された画像には、数字のタグが付けられたオブジェクトや背景の特徴があります。タグ付けされたアイテムを列挙してください。",
    "この画像には、いくつかのオブジェクトやセクションが数字でタグ付けされています。それぞれの番号の背後にあるものを特定し、リストアップできますか？",
    "この画像には、特定の特徴やアイテムに数字のタグが付けられています。それぞれの数値指標の下に何があるのかを詳述してください。",
    "この画像内のさまざまな要素に数字のIDが付けられているのがわかりますか？それぞれの番号がタグ付けしているものをリストアップしてください。",
    "この画像内の特定のオブジェクトや背景部分には数字のIDが付けられています。これらの番号で識別されるものをリストアップできますか？",
    "画像には、さまざまなアイテムや背景の部分に数字のIDが表示されています。これらの数字が何を明らかにしているかをリストアップしてください。",
    "この画像にはアイテムや背景要素に数字のタグが付けられています。これらのタグにより識別された内容を列挙してください。",
    "この画像のアイテムや背景に見える数字のタグを見て、各タグが何を示しているかを区別できますか？",
    "異なるオブジェクトやエリアを強調する数字のIDが付けられた画像を見てください。これらの番号に関連する識別情報をリストアップしてください。",
    "一部のオブジェクト（アイテムや背景）の上に数字のタグが付けられた画像があります。これらのものの名前をリストアップしてください。",
]

INSTRUCTIONS = {
    "labels_ja": None,
}

DATA_PATH = "./dataset/list_items_one_by_one_ja"


def check_abnormal_answer(data, key):
    answer_ja = data[key]
    # answer_en = data[key.replace("_ja", "")].lower()

    curation_list_ja = [
        # "ｔ個",
        # "知りません",
        # "ￒ",
        # "わかりません",
        # "分かりません",
        "ごめんなさい",
        "すみません",
        # "ｒ本",
        # "ｓ匹",
        # "ｒ匹",
        # "ｒつ",
        # "説明はつけない",
        "申し訳",
        "訳され",
    ]

    # curation_list_en = [
    #    #"translat",
    # ]

    for item in curation_list_ja:
        if item in answer_ja:
            return True

    # for item in curation_list_en:
    #    if item in answer_en:
    #        return True

    return False


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
        "team-hatakeyama-phase2/list_items_one_by_one_ja", cache_dir="./cache"
    )

    llava_formats = {key: [] for key in INSTRUCTIONS.keys()}

    for data in dataset["train"]:
        image_filename = str(data["id"]) + "." + data["ext"]
        for key in INSTRUCTIONS.keys():
            object_list = []

            for i, label in enumerate(data[key]):
                object_list.append(f"{i}. {label.strip()}")

            answer = "\n".join(object_list)

            # 不要なデータを除去
            # if check_abnormal_answer(data, key):
            # print(answer)
            # print("===============")
            # continue

            # 文章が長過ぎるデータは除去
            # if len(answer) <= 2000:
            llava_formats[key].append(
                create_llava_format(
                    random.choice(INSTRUCTION_POOLS), answer, image_filename
                )
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
