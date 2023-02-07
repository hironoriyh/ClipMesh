import torch
import clip
import argparse

# import pytorch3d as p3d
import trimesh

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
    FoVOrthographicCameras,
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)


from pytorch3d.structures import join_meshes_as_scene, Meshes
from pytorch3d.io import IO, load_obj, load_objs_as_meshes, save_obj
from PIL import Image
import os
import optuna

torch.set_printoptions(precision=3)

cosine_sim = torch.nn.CosineSimilarity()
device = "cuda"
max_y = 10.0
min_y = -1.0

def cosine_avg(features, targets):
    return -cosine_sim(features, targets).mean()

def save_obj_trimesh(mesh_path, img_path):
    m = trimesh.load(mesh_path)
    im = Image.open(img_path)
    material = trimesh.visual.texture.SimpleMaterial(image=im)
    color_visuals = trimesh.visual.TextureVisuals(uv=m.visual.uv, image=im, material=material)
    mesh = trimesh.Trimesh(vertices=m.vertices, faces=m.faces, visual=color_visuals, validate=True, process=False)
    mesh.export(mesh_path)


def construct_renderer(batch_size = 12, device="cuda"):
    # Get a batch of viewing angles. 
    elev = torch.linspace(-10, 10, batch_size)
    azim = torch.linspace(-90, 90, batch_size)

    R, T = look_at_view_transform(dist=5, elev=elev, azim=azim) #at=(0.0, 0.0, 0.0)
    # T += torch.tensor([0, 0, 0])
    # print(T)

    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
    # cameras = FoVOrthographicCameras(R=R, T=T, device=device)


    raster_settings = RasterizationSettings(
        image_size=224, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    
    bp = BlendParams(sigma=1e-2, background_color=(0.5, 0.5, 0.5))

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights,
            blend_params=bp
        )
    )

    return renderer

def construct_scene(device=device):
    ### const meshes
    # mesh_path = os.path.join("data", 'silk_hat', 'silk_hat.obj')
    # img_path = os.path.join("data", 'silk_hat', 'face_texture_kd.png')

    # save_obj_trimesh(mesh_path, img_path)
    mesh_A = load_objs_as_meshes(["data/face/face.obj"], device=device) 
    mesh_B = load_objs_as_meshes(["data/silk_hat/silk_hat.obj"], device=device) 
    # mesh_A.verts_list()[0] += torch.Tensor([0,-2, 0]).to("cuda") # initial offset the object
    mesh_B.verts_list()[0] += torch.Tensor([0, -3, 0]).to("cuda") # initial offset the object

    # meshes = join_meshes_as_scene([mesh_A, mesh_B])
    return mesh_A, mesh_B


def render_images(offset_trans):
    new_mesh_B = mesh_B.offset_verts(offset_trans.expand(mesh_B.verts_packed().shape).to(device)) # moving face
    # new_mesh_A = mesh_A.offset_verts(offset_trans.expand(mesh_A.verts_packed().shape).to(device))
    meshes = join_meshes_as_scene([mesh_A, new_mesh_B])
    meshes = meshes.extend(batch_size)
    images = renderer(meshes)
    return images

def calc_loss(images):
    images = images[..., :-1].permute(0,3,1,2)
    image_features = model.encode_image(images)
    text_features = model.encode_text(text)
    text_features = text_features.expand(image_features.shape)
    loss = cosine_avg(image_features, text_features) 
    return loss

def objective(trial): ### for optuna
    x = trial.suggest_uniform('x', -0.1, 0.1)
    y = trial.suggest_uniform('y', min_y, max_y)
    z = trial.suggest_uniform('z', -0.1, 0.1)
    images = render_images(torch.tensor([x, y, z])) 
    if(trial.number%int(Nitr/10)==0):
        image_grid(images.cpu().detach().numpy(), rows=2, cols=3, rgb=True, path=os.path.join(directory, "%i.png"%trial.number))
    score = calc_loss(images)    

    print('iter: %i, %1.3f, %1.3f, %1.3f score: %1.3f' % (trial.number, x, y, z, score))
    return score

def opt_autograd(Nitr, directory):
    # y_value = torch.Tensor([0.0, 0.1, 0.0])
    # import ipdb; ipdb.set_trace()

    y_value = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)
    # y_value.requires_grad = True
    # lr = 1e-2
    # optimizer = torch.optim.Adam([y_value], lr=lr)

    lr = 1.0
    optimizer = torch.optim.SGD([y_value], lr=lr)

    list_loss = []
    y_list = []


    for itr in range(Nitr):
        optimizer.zero_grad()
        trans = y_value.clone()
        trans[0] = 0
        trans[2] = 0
        images= render_images(trans) 
        loss = calc_loss(images) #
        list_loss.append(loss.to('cpu').detach().numpy())
        y_list.append(trans[1].to('cpu').detach().numpy())
        print(itr, trans[1].item(), loss.item())
        loss.backward()
        optimizer.step()

    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    plt.scatter(y_list, list_loss)
    plt.savefig(os.path.join(directory, "loss.png"))


def opt_bruteforce(Nitr, directory):

    y_list = torch.linspace(min_y, max_y, Nitr)
    # x, y = torch.meshgrid(x_list, y_list, indexing='xy')

    trans_all = torch.zeros((Nitr, 3))
    trans_all[:, 1] = y_list

    list_loss = []
    loss_min = 100.0
    arg_y = []
    min_itr = 0
    for itr, trans in enumerate(trans_all):
        images= render_images(trans) 
        if(itr%int(Nitr/10)==0):
            image_grid(images.cpu().detach().numpy(), rows=2, cols=3, rgb=True, path=os.path.join(directory, "%i.png"%itr))
        loss = calc_loss(images).to('cpu').detach().numpy()
        list_loss.append(loss)
        # import ipdb; ipdb.set_trace()
        if(loss < loss_min):
            loss_min = loss
            print(itr, trans[1],  loss)
            arg_y = trans
            min_itr = itr
        # print("trans: ", trans.numpy(), 'loss: %1.3f' % (loss))

    print("min itr", min_itr, "min trans: ", arg_y, "loss: ", loss_min)

    images= render_images(arg_y) 
    image_grid(images.cpu().detach().numpy(), rows=2, cols=3, rgb=True, path=os.path.join(directory, "best_%f_%i.png"%(arg_y[1], min_itr)))

    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    plt.plot(y_list, list_loss)
    plt.savefig(os.path.join(directory, "loss.png"))


parser = argparse.ArgumentParser()

parser.add_argument("-bf", "--bruteforce",     help='bruteforce loss value check for comparison', action='store_true')
parser.add_argument("-ag", "--autograd",        help='autograd optimization', action='store_true')
parser.add_argument("-ot", "--optuna",          help='optuna optimization', action='store_true')
parser.add_argument("--testimage", action='store_false')

args = parser.parse_args()

###### clip loop #####

model, preprocess = clip.load("ViT-B/32", device=device) # model = model.zero_grad() didnt work

query = "An image of a head wearing a black silk hat deeply"
# query = "An image of black silk hat on top of a head"
# query = "An image of head and black hat far away separated"

# query = "an image of two separated objects on white background"
text = clip.tokenize([query]).to(device)

# set render, meshes
Nitr = 200
batch_size = 12
renderer = construct_renderer(batch_size=batch_size)
mesh_A, mesh_B = construct_scene(device=device)

#### bruteforce
if(args.bruteforce):
    directory = os.path.join("trans_images", query.replace(" ", "_"), "bruteforce")
    os.makedirs(directory, exist_ok = True)
    opt_bruteforce(Nitr, directory)

### optimize with autograd
if(args.autograd):
    directory = os.path.join("trans_images", query.replace(" ", "_"), "autograd")
    os.makedirs(directory, exist_ok = True)
    opt_autograd(Nitr, directory)

#### optuna 
if(args.optuna):
    directory = os.path.join("trans_images", query.replace(" ", "_"), "optuna")
    os.makedirs(directory, exist_ok = True)

    study = optuna.create_study(direction="minimize") # 最適化処理を管理するstudyオブジェクト
    study.optimize(objective, # 目的関数
                n_trials=Nitr # トライアル数
                )

    images = render_images(torch.tensor([study.best_params["x"], study.best_params["y"], study.best_params["z"]])) 
    image_grid(images.cpu().detach().numpy(), rows=2, cols=3, rgb=True, path=os.path.join(directory, "best_%f.png"%study.best_params["y"]))
    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    plt.scatter([trial.params["y"] for trial in study.trials], [trial.value for trial in study.trials])
    plt.savefig(os.path.join(directory, "loss.png"))

if (args.testimage ):
    # directory = 
    images = render_images(torch.tensor([0,0,0])) 
    # import ipdb; ipdb.set_trace()
    calc_loss(images)
    # image_grid(images.cpu().detach().numpy(), rows=2, cols=3, rgb=True, path=os.path.join(directory, "best_%f.png"%study.best_params["y"]))

# optuna.visualization.matplotlib.plot_optimization_history(study)
# plt.tight_layout()
# path = os.path.join(directory, "optuna_plot.png")
# plt.savefig(path)
# print("best", study.best_value)