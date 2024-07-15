#!/bin/bash

# 以下のデータセットを用意
# commoncatalog-cc-by-recap-diverse_questions

mkdir ./dataset/commoncatalog-cc-by-recap-diverse_questions

### キャプション準備
python tools/commoncatalog-cc-by-recap-diverse_questions/to_llava_format.py
