# list_items_one_by_one_ja
## jsonファイルの準備
以下を実行
```
mkdir ./dataset/list_items_one_by_one_ja
python tools/list_items_one_by_one_ja/to_llava_format_detailed.py
python tools/list_items_one_by_one_ja/to_llava_format_labels.py
```

## 画像の準備
v4_stage_2.jsonでは、https://github.com/hibikaze-git/create-data-for-vlm でデータを合成した際に出力された画像のディレクトリを指定しています。データの合成を行なわない場合は、別途画像を準備する必要があります。

### list_items_one_by_one_ja_detailed
- 以下を参考に、MS-COCO 2017の画像をダウンロードします
- list_items_one_by_one_ja_detailedのimage_folderに指定します
```
#!/bin/bash

wget -P . http://images.cocodataset.org/zips/train2017.zip
unzip ./train2017.zip
```

### list_items_one_by_one_ja_labels
- MS-COCO 2017の画像に写っているオブジェクトに番号を振った新しい画像を作成する必要があります
- 以下のコードを参考に、team-hatakeyama-phase2/list_items_one_by_one_jaのcentersカラムの位置に0から始まる番号を振ってください
https://github.com/hibikaze-git/create-data-for-vlm/blob/main/datasets/list_items_one_by_one/florence.py#L55-L120
- 番号を振った画像が保存されたディレクトリをlist_items_one_by_one_ja_labelsのimage_folderに指定してください
