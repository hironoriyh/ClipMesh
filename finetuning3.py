import json
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import clip
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')  # 'Agg' はバックエンドの1つで、GUIなしでの使用をサポートしています
import matplotlib.pyplot as plt

# 学習曲線を保存するためのリスト
train_losses = []

# データのパス
json_path = '/home/itoh/ClipMesh/3DCoMPaT-v2/loaders/3D/images_template_d_p.json'

# JSONファイルからデータを読み込む関数
def load_data_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

# データを読み込む
images_data = load_data_from_json(json_path)["images_data"]
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

for item in images_data:
  img_path = item['image_path']
  caption = item['caption'][:77]
  stable_list_image_path.append(img_path)
  stable_list_txt.append(caption)

# print(list_txt)

# データセットの作成
images_dataset = image_title_dataset(stable_list_image_path, stable_list_txt)

# データローダーの設定
# batch_size = 300

# ファインチューニングの設定
optimizer = torch.optim.Adam(model.parameters(), lr=1e-6,betas=(0.9,0.98),eps=1e-6,weight_decay=0.2) # the lr is smaller, more safe for 
# Specify the loss function
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# ファインチューニングのエポック数
num_epochs = 60

train_dataloader = DataLoader(images_dataset, batch_size=100, shuffle=True)

for epoch in range(num_epochs):
    pbar = tqdm(train_dataloader, total=len(train_dataloader))
    # 安定した椅子のデータでファインチューニング
    for batch in pbar:
        optimizer.zero_grad()

        images,texts = batch 

        # print(images)
        images = images.to(device)
        # print(texts)
        texts = texts.to(device)

        
        logits_per_image, logits_per_text = model(images, texts)

        # Compute loss
        ground_truth = torch.arange(len(images),dtype=torch.long,device=device)
        # print(len(images))
        loss = (loss_img(logits_per_image,ground_truth) + loss_txt(logits_per_text,ground_truth))/2

        loss.backward()

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)
            optimizer.step()
            clip.model.convert_weights(model)

        pbar.set_description(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")

    # エポックごとに損失を保存
    train_losses.append(loss.item())

# 学習曲線のグラフを描画して保存
plt.plot(range(1, num_epochs+1 ), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.savefig('training_loss_curve_1e-6_60e.png')

# ファインチューニングが完了したらモデルを保存
torch.save(model.state_dict(), "all_3_d_p_1e-6_60e.pth")
