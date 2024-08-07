# wit_base
## 元となるデータの準備
- 以下を用いて画像とjsonl(データごと)を準備してください  
https://github.com/hibikaze-git/create-data-for-vlm/tree/main/datasets/wit
- jsonl（データごと）を以下を参考に1つのjsonlにまとめます  
https://github.com/hibikaze-git/create-data-for-vlm/blob/main/tools/merge_jsonl_files.sh
- まとめたjsonlを「tools/wit_base/wit_base.jsonl」にコピーします

## 学習データの準備
以下を実行
```
mkdir ./dataset/wit_base
python tools/wit_base/to_llava_format_fix_instruction.py
```
