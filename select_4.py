import clip
import os
from PIL import Image
import torch

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIPモデルと前処理の読み込み
model, preprocess = clip.load("ViT-B/32", device=device)

target_image_path = "/Users/itouseiji/Desktop/3dcompat-v2/3DCoMPaT-v2/loaders/3D/output_images_200/scene_0.png"
# 入力テキスト
input_text = "a photo of a cat"
text_input = clip.tokenize([input_text]).to(device)
text_feature = model.encode_text(text_input)

# 画像のフォルダパス
image_folder_path = "/Users/itouseiji/Desktop/3dcompat-v2/3DCoMPaT-v2/loaders/3D/output_images"

# テキストファイルへの出力パス
output_file_path = f"/Users/itouseiji/Desktop/3dcompat-v2/3DCoMPaT-v2/loaders/3D/p_chair.txt"

# テキストファイルを開いて初期化
with open(output_file_path, 'w') as output_file:
    output_file.write(f"Input Text: p_chair\n\n")
    output_file.write("Image File\tSimilarity Score\n")

results = []  # 結果を格納するリストを初期化

target_image = preprocess(Image.open(target_image_path)).unsqueeze(0).to(device)
target_image_feature = model.encode_image(target_image)

for filename in os.listdir(image_folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder_path, filename)

        # 画像の前処理とエンコード
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_feature = model.encode_image(image)

        # テキストのエンコード
        # text_input = clip.tokenize([input_text]).to(device)
        # text_feature = model.encode_text(text_input)

        # スコアを計算
        score = torch.cosine_similarity(image_feature, text_feature).item()

        # 結果をリストに追加
        results.append((filename, score))

# 類似度スコアで降順にソート
results = sorted(results, key=lambda x: x[1], reverse=True)

# テキストファイルに保存
with open(output_file_path, 'w') as output_file:
    output_file.write(f"Input Text: p_chair\n\n")
    output_file.write("Image File\tSimilarity Score\n")
    
    for filename, score in results:
        output_file.write(f"{filename}\t{score}\n")

print("Similarity scores saved to:", output_file_path)