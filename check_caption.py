import json
from PIL import Image

#json_path = "./dataset/llava_pretrain_stair.json"
json_path = "./dataset/llava_visual_genome_ja.json"

# JSONファイルの読み込み
with open(json_path, "r") as file:
    data = json.load(file)

print(data[:5])

index = 2000

# 画像とキャプションの取得
image_path = data[index].get("image")
caption = data[index].get("conversations", "No caption available")

#image_dir = "./dataset/images/stage1/MS-COCO-train2014/"
image_dir = "./dataset/images/stage2/VG_100K_2/"

# 画像の完全なパスを作成
full_image_path = image_dir + image_path

# 画像とキャプションの表示
print("Image Path:", full_image_path)

# 画像の表示
try:
    img = Image.open(full_image_path)
    img.show()  # これにより、デフォルトの画像ビューアで画像が表示されます
except FileNotFoundError:
    print(f"Image not found: {full_image_path}")

# キャプションの表示
if isinstance(caption, list):
    for line in caption:
        print(f"Caption: {line}")
else:
    print(f"Caption: {caption}")
