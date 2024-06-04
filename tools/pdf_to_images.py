"""
python pdf_to_images.py ./data ./data/output --num_processes 8
python pdf_to_images.py ./data/public_document00152.pdf ./data/output
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import fitz
from tqdm import tqdm


def get_file_paths(directory_or_file, extension=None):
    """指定したディレクトリ直下のファイルまたは単一のファイルパスのリストを取得

    Args:
        directory_or_file (str): ディレクトリまたはファイルのパス
        extension (str, optional): ファイル拡張子（例: '.pdf'）

    Returns:
        list: ファイルパスのリスト
    """
    path = Path(directory_or_file).resolve()
    if path.is_dir():
        if extension:
            return [str(file) for file in path.glob(f"*{extension}")]
        else:
            return [str(file) for file in path.iterdir() if file.is_file()]
    elif path.is_file():
        if extension and path.suffix != extension:
            return []
        return [str(path)]
    else:
        return []


def process_pdf(file_path, output_folder):
    """PDFファイルのページを画像ファイルに変換する

    Args:
        file_path (str): PDFファイルのパス
        output_folder (str): 画像ファイルを出力するフォルダ
    """
    output_folder_path = Path(output_folder)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    path_file = Path(file_path)
    base_file_name = path_file.stem

    # PDFファイルを開く
    print(f"Processing {file_path}")

    pdf = fitz.open(file_path)

    # 1ページずつ画像ファイルとして保存する
    for i, page in enumerate(pdf):
        pix = page.get_pixmap()
        image_f_name = f"{base_file_name}_{str(i+1)}.jpg"
        image_f_path = output_folder_path / image_f_name
        pix.save(image_f_path)


def main(file_path_list, output_folder, num_processes):
    """PDFファイルのページを画像ファイルに変換する

    Args:
        file_path_list (list): PDFファイルパス名のリスト
        output_folder (str): 画像ファイルを出力するフォルダ
        num_processes (int): 使用するプロセスの数
    """
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = [
            executor.submit(process_pdf, file_path, output_folder)
            for file_path in file_path_list
        ]

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing PDFs"
        ):
            future.result()  # ここで例外が発生した場合は再度スローされます


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PDF pages to image files.")
    parser.add_argument(
        "input_folder_or_file",
        type=str,
        help="Input folder containing PDF files or a single PDF file",
    )
    parser.add_argument(
        "output_folder", type=str, help="Output folder to save image files"
    )
    parser.add_argument(
        "--num_processes", type=int, default=4, help="Number of processes to use"
    )

    args = parser.parse_args()

    pdf_file_paths = get_file_paths(args.input_folder_or_file, extension=".pdf")
    main(pdf_file_paths, args.output_folder, args.num_processes)
