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
from plot_image_grid import image_grid

cosine_sim = torch.nn.CosineSimilarity()

def cosine_avg(features, targets):
    return -cosine_sim(features, targets).mean()

mesh_path = os.path.join("data", 'playcard', 'playcard.obj')
img_path = os.path.join("data", 'playcard', 'playcard_texture_kd.png')

# m = trimesh.load(mesh_path)
# im = Image.open(img_path)

# material = trimesh.visual.texture.SimpleMaterial(image=im)
# color_visuals = trimesh.visual.TextureVisuals(uv=m.visual.uv, image=im, material=material)
# mesh = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, visual=color_visuals, validate=True, process=False)
# mesh.export(mesh_path)


mesh_A = load_objs_as_meshes(["data/playcard/playcard.obj"])
mesh_B = load_objs_as_meshes(["data/face/face.obj"])

R, T = look_at_view_transform(8.0, 45.0, 180) 
cameras = FoVPerspectiveCameras(R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=224, 
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

# num_views = 5
# meshes = meshes.extend(num_views)
# target_images = renderer(meshes, cameras=cameras, lights=lights)

###### clip #####

# モデルの読み込み
device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)
model, preprocess = clip.load("ViT-B/32", device=device)

text = clip.tokenize(["a face with a silkhut"]).to(device)

offset_trans = torch.zeros((1,3), requires_grad=True)
optimizer = torch.optim.SGD([offset_trans], lr=1.0, momentum=0.9)
# optimizer = torch.optim.Adam([offset_trans], lr=0.05)

for i in range (100):

    new_mesh_A = mesh_A.offset_verts(offset_trans.expand(mesh_A.verts_packed().shape))
    meshes = join_meshes_as_scene([new_mesh_A, mesh_B])

    image = renderer(meshes)
    # import ipdb; ipdb.set_trace()

    if(i%20==0):
        plt.figure(figsize=(10, 10))
        plt.imsave( "trans_images/%i.png"%i, image[0, ..., :3].detach().numpy())

    image = image[:,:,:,:3] # 
    image = image.permute(0,3,1,2).to(device) # torch.Size([1, 512, 512, 3]) -> torch.Size([1, 3, 512, 512])
    
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    clip_loss = cosine_avg(image_features, text_features)

    # 推論
    # logits_per_image, logits_per_text = model(image, text)
    optimizer.zero_grad()
    clip_loss.backward()
    optimizer.step()

    print(clip_loss, offset_trans)
    # offset_trans += torch.Tensor((0.1, 0, 0))


        # probs = logits_per_image.softmax(dim=-1).cpu().numpy()

