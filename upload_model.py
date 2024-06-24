"""
モデルのアップロード

python upload_model.py ./path/to/local/folder your-username/your-repo-id
"""

import argparse

from huggingface_hub import HfApi


def upload_to_hf_hub(local_folder_path, repo_id):
    # APIインスタンスの作成
    api = HfApi()

    # アップロード先のリポジトリ情報
    repo_type = "model"  # リポジトリの種類（"model", "dataset", "space"など）
    commit_message = "upload model"  # コミットメッセージ

    # アップロード処理
    api.upload_folder(
        folder_path=local_folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
        ignore_patterns=[
            "runs",
            "log",
            "training_args.bin",
            "global_step*",
            "latest",
            "rng_state.pth",
            "rng_state_*.pth",
            "scheduler.pt",
            "trainer_state.json",
            "zero_to_fp32.py",
            "optimizer.pt",
            ".huggingface",
            ".gitattributes",
            "checkpoint-*",
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload a local folder to Hugging Face Hub."
    )
    parser.add_argument(
        "local_folder_path", type=str, help="The path of the local folder to upload."
    )
    parser.add_argument(
        "repo_id", type=str, help="The repository ID on Hugging Face Hub."
    )

    args = parser.parse_args()

    upload_to_hf_hub(args.local_folder_path, args.repo_id)
