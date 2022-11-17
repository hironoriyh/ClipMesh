import torch
import clip
from PIL import Image

import pytorch3d as p3d
import trimesh
import torch

import matplotlib.pyplot as plt

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

from pytorch3d.structures import join_meshes_as_scene, Meshes
from pytorch3d.io import IO, load_obj, load_objs_as_meshes, save_obj
from PIL import Image
import os

mesh_path = os.path.join("data", 'playcard', 'playcard.obj')
img_path = os.path.join("data", 'playcard', 'playcard_texture_kd.png')

m = trimesh.load(mesh_path)
im = Image.open(img_path)

material = trimesh.visual.texture.SimpleMaterial(image=im)
color_visuals = trimesh.visual.TextureVisuals(uv=m.visual.uv, image=im, material=material)
mesh = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, visual=color_visuals, validate=True, process=False)
mesh.export(mesh_path)


mesh_A = load_objs_as_meshes(["data/playcard/playcard.obj"])
mesh_B = load_objs_as_meshes(["data/face/face.obj"])
meshes_list = [mesh_A, mesh_B] #"data/playcard/playcard.obj"]

print(mesh_A.verts_list()[0])
mesh_A.verts_list()[0] +=  torch.Tensor([3,0,0])
print(mesh_A.verts_list()[0])

meshes = join_meshes_as_scene(meshes_list)

device = 0

R, T = look_at_view_transform(8.0, 45.0, 180) 
cameras = FoVPerspectiveCameras(R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=512, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)


lights = PointLights(location=[[0.0, 0.0, -3.0]])

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        # device=device, 
        cameras=cameras,
        lights=lights
    )
)

images = renderer(meshes)
plt.figure(figsize=(10, 10))
plt.imsave( "test.png", images[0, ..., :3].cpu().numpy())
plt.axis("off");


# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

# モデルの読み込み
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
model, preprocess = clip.load("ViT-B/32", device=device)

# # 画像とテキストの準備

image = preprocess(Image.open("test.png")).unsqueeze(0).to(device)
text = clip.tokenize(["a man", "a dog", "a cat"]).to(device)

with torch.no_grad():
    # 画像とテキストのエンコード
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    # 推論
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# 類似率の出力
print("Label probs:", probs)