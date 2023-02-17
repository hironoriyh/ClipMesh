import torch
import clip
from PIL import Image

import pytorch3d as p3d
import trimesh

import optuna
from optuna.visualization import plot_optimization_history
import matplotlib.pyplot as plt

import numpy as np
from plot_image_grid import image_grid


# Data structures and functions for rendering
from pytorch3d.structures import Meshes
# from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
# from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
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
from plot_image_grid import image_grid

cosine_sim = torch.nn.CosineSimilarity()

def cosine_avg(features, targets):
    return -cosine_sim(features, targets).mean()

def save_obj_trimesh(mesh_path, img_path):
    m = trimesh.load(mesh_path)
    im = Image.open(img_path)
    material = trimesh.visual.texture.SimpleMaterial(image=im)
    color_visuals = trimesh.visual.TextureVisuals(uv=m.visual.uv, image=im, material=material)
    mesh = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, visual=color_visuals, validate=True, process=False)
    mesh.export(mesh_path)


mesh_path = os.path.join("data", 'silk_hat', 'silk_hat.obj')
img_path = os.path.join("data", 'silk_hat', 'face_texture_kd.png')

# save_obj_trimesh(mesh_path, img_path)

device = "cuda"

mesh_A = load_objs_as_meshes(["data/face/face.obj"], device=device)
mesh_B = load_objs_as_meshes(["data/silk_hat/silk_hat.obj"], device=device)
mesh_A.verts_list()[0] += torch.Tensor([0,-2, 0]).to("cuda") # initial offset the object
mesh_B.verts_list()[0] += torch.Tensor([0, 2, 0]).to("cuda") # initial offset the object

meshes = join_meshes_as_scene([mesh_A, mesh_B])

batch_size = 10

# Get a batch of viewing angles. 
elev = torch.linspace(0, 60, batch_size)
azim = torch.linspace(-180, 180, batch_size)

# R, T = look_at_view_transform(8.0, 45.0, 180) 
R, T = look_at_view_transform(dist=8, elev=elev, azim=azim)
cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

raster_settings = RasterizationSettings(
    image_size=224, 
    blur_radius=0.0, 
    faces_per_pixel=1, 
)


lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras, 
        raster_settings=raster_settings
    ),
    shader=SoftPhongShader(
        device=device, 
        cameras=cameras,
        lights=lights
    )
)
###### clip loop #####

# print(device)
model, preprocess = clip.load("ViT-B/32", device=device)
# model = model.zero_grad()

query = "The face of a man wearing a silk hat"
# query = "an image of two separated objects on white background"
text = clip.tokenize([query]).to(device)

os.makedirs(os.path.join("trans_images", query.replace(" ", "_")), exist_ok = True)

list_loss = []

# plt.close()
# plt.clf()
fig = plt.figure()

# s1_params = 1.0
Nitr = 300

def render_images(x, y, z):
    offset_trans = torch.tensor([x, y, z])
    new_mesh_A = mesh_A.offset_verts(offset_trans.expand(mesh_A.verts_packed().shape).to(device))
    meshes = join_meshes_as_scene([new_mesh_A, mesh_B])
    meshes = meshes.extend(batch_size)
    images = renderer(meshes)
    return images

def objective(trial):
    x = trial.suggest_uniform('x', -3, 3)
    y = trial.suggest_uniform('y', -3, 3)
    z = trial.suggest_uniform('z', -3, 3)
    images = render_images(0, y, 0) 
    if(trial.number%int(Nitr/10)==0):
        image_grid(images.cpu().detach().numpy(), rows=4, cols=5, rgb=True, path=os.path.join("trans_images", query.replace(" ", "_"), "%i.png"%trial.number))
    
    images = images[..., :-1].permute(0,3,1,2)
    # print(images.shape)
    image_features = model.encode_image(images)
    text_features = model.encode_text(text)
    text_features = text_features.expand(image_features.shape)

    score = cosine_avg(image_features, text_features)    
    # list_loss.append(score.item())

    print('iter: %i, %1.3f, %1.3f, %1.3f score: %1.3f' % (trial.number, x, y, z, score))
    return score

study = optuna.create_study(direction="minimize") # 最適化処理を管理するstudyオブジェクト
study.optimize(objective, # 目的関数
               n_trials=Nitr # トライアル数
              )
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()
path = os.path.join("trans_images", query.replace(" ", "_"), "optuna_plot.png")
plt.savefig(path)
print("best", study.best_value)

# import ipdb; ipdb.set_trace()
images = render_images(0, study.best_value.y, 0) 
image_grid(images.cpu().detach().numpy(), rows=4, cols=5, rgb=True, path=os.path.join("trans_images", query.replace(" ", "_"), "best.png"))
    
# print()
# fig = plt.figure()
# plt.figure(figsize=(10, 10))
# plt.plot(list_loss)
# plt.savefig(os.path.join("trans_images", query.replace(" ", "_"), "loss.png"))

        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

