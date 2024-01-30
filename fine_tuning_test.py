import clip
import os
from PIL import Image
import torch

# デバイスの設定
device = "cuda" if torch.cuda.is_available() else "cpu"

# CLIPモデルと前処理の読み込み
model, preprocess = clip.load("ViT-B/32", device=device)

# model.load_state_dict(torch.load("lr_1e-6.pth"))
# model.load_state_dict(torch.load("/home/itoh/ClipMesh/fine_tuned_crossentropy_contrastive_model.pth"))
# model.load_state_dict(torch.load("/home/itoh/ClipMesh/fine_tuned_model.pth"))
# model.load_state_dict(torch.load("/home/itoh/ClipMesh/fine_tuned_crossentropy_contrastive_model.pth"))
# model.load_state_dict(torch.load("/home/itoh/ClipMesh/finetuning2.pth"))
model.load_state_dict(torch.load("/home/itoh/ClipMesh/finetuning_models/all_3_1e-6/all_3.pth"))

# 入力テキスト
input_text = "A photo of stable chair"
input_text = "A photo of unstable chair"
# input_text = "A photo of unstable  detach chair"
# input_text = "A photo of unstable penetrate chair"
# input_text = "A photo of stable object"

# 画像のフォルダパス
image_folder_path = "/home/itoh/ClipMesh/fine_tuning_test_data/test2/"

# テキストファイルへの出力パス
output_file_path = f"/home/itoh/ClipMesh/finetuning_models/all_3_1e-6/{input_text}.txt"

# 画像フォルダ内の画像に対して類似度を計算
results = []  # To store results for sorting

for filename in os.listdir(image_folder_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(image_folder_path, filename)

        # 画像の前処理とエンコード
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_feature = model.encode_image(image)

        # テキストのエンコード
        text_input = clip.tokenize([input_text]).to(device)
        text_feature = model.encode_text(text_input)

        # スコアを計算
        score = torch.cosine_similarity(image_feature, text_feature).item()

        # 結果を保存
        results.append((filename, score))

# 結果をスコアが高い順にソート
results.sort(key=lambda x: x[1], reverse=True)

# テキストファイルを開いて初期化
with open(output_file_path, 'w') as output_file:
    output_file.write(f"Input Text: {input_text}\n\n")
    output_file.write("Image File\tSimilarity Score\n")

    # 並べ替えた結果をテキストファイルに追記
    for filename, score in results:
        output_file.write(f"{filename}\t{score}\n")

print("Similarity scores saved to:", output_file_path)