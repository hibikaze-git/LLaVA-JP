from datasets import load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    img_dataset = load_dataset(
        "common-canvas/commoncatalog-cc-by",
        split="train",
        streaming=True,
        cache_dir="./cache",
    )

    for i, data in tqdm(enumerate(img_dataset), total=300_000):
        data["jpg"].save(
            f"dataset/commoncatalog-cc-by/cc_300k_photoid/{data['photoid']}.{data['ext']}"
        )

        if i >= 300_000 - 1:
            break

    print("========== finished ==========")
