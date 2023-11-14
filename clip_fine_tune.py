import json
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import clip

# データのパス
stable_json_path = '/home/itoh/ClipMesh/3DCoMPaT-v2/loaders/3D/stable_data.json'
unstable_json_path = '/home/itoh/ClipMesh/3DCoMPaT-v2/loaders/3D/unstable_data.json'

# JSONファイルからデータを読み込む関数
def load_data_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# データを読み込む
stable_data = load_data_from_json(stable_json_path)["stable_images"]
unstable_data = load_data_from_json(unstable_json_path)["unstable_images"]
# print(type(stable_data))

# CLIPモデルの読み込み
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# カスタムデータセットの作成
class image_title_dataset():
    def __init__(self, list_image_path,list_txt):
        # Initialize image paths and corresponding texts
        self.image_path = list_image_path
        # Tokenize text using CLIP's tokenizer
        self.title  = clip.tokenize(list_txt)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        # Preprocess image using CLIP's preprocessing function
        image = preprocess(Image.open(self.image_path[idx]))
        title = self.title[idx]
        return image, title
    
# Function to convert model's parameters to FP32 format
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        if p.grad is not None:  # 勾配が存在する場合のみ変換を行う
            p.data = p.data.float() 
            p.grad.data = p.grad.data.float()


if device == "cpu":
  model.float()

    
stable_list_image_path = []
stable_list_txt = []
unstable_list_image_path = []
unstable_list_txt = []

for item in stable_data:
  img_path = item['image_path']
  caption = item['caption'][:50]
  stable_list_image_path.append(img_path)
  stable_list_txt.append(caption)

for item in unstable_data:
  img_path = item['image_path']
  caption = item['caption'][:50]
  unstable_list_image_path.append(img_path)
  unstable_list_txt.append(caption)

# print(list_txt)

# データセットの作成
stable_dataset = image_title_dataset(stable_list_image_path, stable_list_txt)
unstable_dataset = image_title_dataset(unstable_list_image_path, unstable_list_txt)

# データローダーの設定
batch_size = 5

# ファインチューニングの設定
optimizer = torch.optim.Adam(model.parameters(),1e-6)

# ファインチューニングのエポック数
num_epochs = 10

for epoch in range(num_epochs):
    stable_data_loader = torch.utils.data.DataLoader(stable_dataset, batch_size=batch_size, shuffle=True)
    unstable_data_loader = torch.utils.data.DataLoader(unstable_dataset, batch_size=batch_size, shuffle=True)
    # 安定した椅子のデータでファインチューニング
    for images, texts in stable_data_loader:
        # print(images)
        images = images.to(device)
        # print(texts)
        texts = texts.to(device)

        optimizer.zero_grad()
        logits_per_image, logits_per_text = model(images, texts)
        # print(logits_per_image)
        # print(logits_per_text)
        # loss = (logits_per_image - logits_per_text).mean()
        # loss = 1 - (logits_per_image * logits_per_text).sum(dim=-1).mean()

        # モデルから画像とテキストのエンコードを取得
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        # コサイン類似度損失の計算
        similarity = F.cosine_similarity(image_features, text_features)
        loss = 1-similarity.mean()

        loss.backward()

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

    # 不安定な椅子のデータでファインチューニング
    for images, texts in unstable_data_loader:
        images = images.to(device)
        texts = texts.to(device)

        optimizer.zero_grad()
        # logits_per_image, logits_per_text = model(images, texts)
        # loss = (logits_per_image - logits_per_text).mean()
        # # loss = 1 - (logits_per_image * logits_per_text).sum(dim=-1).mean()

        # モデルから画像とテキストのエンコードを取得
        image_features = model.encode_image(images)
        text_features = model.encode_text(texts)

        # コサイン類似度損失の計算
        similarity = F.cosine_similarity(image_features, text_features)
        loss = 1 - similarity.mean()
        # print(loss)

        loss.backward()
        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)


    print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

# ファインチューニングが完了したらモデルを保存
torch.save(model.state_dict(), "fine_tuned_model.pth")
