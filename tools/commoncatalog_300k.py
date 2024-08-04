import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    cc_300k_dataset = load_dataset("alfredplpl/commoncatalog-cc-by-ja", split="train")
    cc_300k_dataset.to_csv("dataset/commoncatalog-cc-by/cc_300k_dataset.csv")
    cc_300k_df = pd.read_csv("dataset/commoncatalog-cc-by/cc_300k_dataset.csv")
    img_dataset = load_dataset(
        "common-canvas/commoncatalog-cc-by", split="train", streaming=True
    )
    data_info = []

    for i, data in tqdm(enumerate(img_dataset), total=300_000):
        data["jpg"].save(f"dataset/commoncatalog-cc-by/cc_300k/{i:09}.jpg")
        no = cc_300k_df[cc_300k_df["photoid"] == data["photoid"]]["photoid"].index
        data_info.append(
            {
                # "height": height,
                # "width": width,
                # "ratio": ratio,
                "path": f"{i:09}.jpg",
                "caption": cc_300k_df["brief_caption"][no],
                "id": data["photoid"],
            }
        )
        if i >= 300_000 - 1:
            break
    print(f"length of data_info: {len(data_info)}")

    cc_300k_llava_formats = []

    for i in tqdm(range(len(data_info)), total=len(data_info)):
        captions = (data_info[i]["id"], data_info[i]["caption"].item())

        if len(captions) < 2:
            print("not found caption")
            break

        id, caption = captions
        llava_format = {}
        conversations = []

        conversation_user = {
            "from": "ユーザー",
            "value": "画像について説明してください。\n<image>",
        }
        conversation_system = {"from": "システム", "value": caption}
        conversations.append(conversation_user)
        conversations.append(conversation_system)

        llava_format["id"] = id
        llava_format["image"] = data_info[i]["path"]
        llava_format["conversations"] = conversations

        cc_300k_llava_formats.append(llava_format)

    print(f"length of cc_300k_llava_formats: {len(cc_300k_llava_formats)}")
    chat_ja_path = Path(
        "dataset", "commoncatalog-cc-by", "llava_cc_300k_caption.json"
    )
    with open(chat_ja_path, mode="w") as f:
        json.dump(cc_300k_llava_formats, f, indent=2, ensure_ascii=False)
