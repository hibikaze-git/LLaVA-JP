# commoncatalog-cc-by-sa-ja-complex
## jsonファイルの準備
以下を実行
```
mkdir ./dataset/commoncatalog-cc-by-sa-ja-complex
python tools/commoncatalog-cc-by-sa-ja-complex/to_llava_format_curation_calm3.py
```
## 画像の準備
v4_stage_2.jsonでは、https://github.com/hibikaze-git/create-data-for-vlm でデータを合成した際に出力された画像のディレクトリを指定しています。データの合成を行なわない場合は、別途画像を準備する必要があります。
- [こちら](https://huggingface.co/datasets/common-canvas/commoncatalog-cc-by-sa/tree/main)のデータセットの4をダウンロード
- load_dataset等で各parquetファイルを読み込み、画像をphotoid.ext（例: 457985375.jpg）の形式で保存する
- 画像が保存されているディレクトリをimage_folderに指定する(dataset/commoncatalog-cc-by-sa-complex/images 等)
