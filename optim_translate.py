import torch
import clip
import argparse

# import pytorch3d as p3d
import trimesh

import matplotlib.pyplot as plt
import numpy as np
from plot_image_grid import image_grid

from render_nvdiff import const_cfg, nvdiff_meshes, create_cams, nvdiff_rendermeshes, const_scene_camparams, const_trainrender

from PIL import Image
import os
import optuna

import torchvision


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


def render_images(trans, render_meshes, cams, cfg):

    # render_meshes_original = render_meshes.copy().deepcopy()

    # add diff of trans to the meshes
    # for mesh in render_meshes:
    render_meshes[1].v_pos = render_meshes[1].v_pos.clone().detach() + trans.to(device)
    complete_scene, params_camera  = const_scene_camparams(render_meshes, cams)
    images = const_trainrender(complete_scene, params_camera, cfg)

    # bring back mesh positions
    # for mesh in render_meshes:
    render_meshes[1].v_pos = render_meshes[1].v_pos.clone().detach() -  trans.to(device)

    return images

def calc_loss(images):
    # images = images[..., :-1].permute(0,3,1,2)
    image_features = model.encode_image(images)
    text_features = model.encode_text(text)
    text_features = text_features.expand(image_features.shape)
    loss = cosine_avg(image_features, text_features) 
    return loss


def saveimgs(images, imgpath="testnvdiff.png"):
    # s_log = images[torch.randint(low=0, high=cfg["batch_size"], size=(5 if cfg["batch_size"] > 5 else cfg["batch_size"], )) , :, :, :]
    s_log = torchvision.utils.make_grid(images)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    ndarr = s_log.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(imgpath)


def opt_bruteforce(Nitr, directory):

    print("start bruteforce between %f and %f"%(min_y, max_y))

    y_list = torch.linspace(min_y, max_y, Nitr)
    # x, y = torch.meshgrid(x_list, y_list, indexing='xy')

    trans_all = torch.zeros((Nitr, 3))
    trans_all[:, 1] = y_list

    list_loss = []
    loss_min = 100.0
    arg_y = []
    min_itr = 0

    cfg = const_cfg()
    cams = create_cams(cfg)
    meshes, subdiv, _, _ = nvdiff_meshes(cfg)
    render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)

    for itr, trans in enumerate(trans_all):
        # render_meshes
        images= render_images(trans, render_meshes, cams, cfg) 
        if(itr%int(Nitr/10)==0):
            saveimgs(images, imgpath=os.path.join(directory, "%i.png"%itr))
        loss = calc_loss(images).to('cpu').detach().numpy()
        list_loss.append(loss)
        # import ipdb; ipdb.set_trace()
        if(loss < loss_min):
            loss_min = loss
            print("updated at: ", itr, trans[1],  loss)
            arg_y = trans
            min_itr = itr

    print("min itr", min_itr, "min trans: ", arg_y, "loss: ", loss_min)
    images= render_images(arg_y, render_meshes, cams, cfg) 
 
    saveimgs(images, imgpath=os.path.join(directory, "opt_min_at_%i.png"%min_itr))

    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    plt.plot(y_list, list_loss)
    plt.savefig(os.path.join(directory, "loss.png"))

def opt_autograd(Nitr, directory):

    y_value = torch.tensor([0.0, 1.0, 0.0], requires_grad=True)

    lr = 1.0
    optimizer = torch.optim.SGD([y_value], lr=lr)

    list_loss = []
    y_list = []


    cfg = const_cfg()
    cams = create_cams(cfg)
    meshes, subdiv, _, _ = nvdiff_meshes(cfg)
    render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)


    for itr in range(Nitr):
        optimizer.zero_grad()
        trans = y_value.clone()
        trans[0] = 0 # forced to set the x value to 0
        trans[2] = 0 # forced to set the z value to 0
        images= render_images(trans, render_meshes, cams, cfg) 
        loss = calc_loss(images) #
        list_loss.append(loss.to('cpu').detach().numpy())
        y_list.append(trans[1].to('cpu').detach().numpy())
        print(itr, "trans y: %0.4f, loss: %0.4f"%(trans[1].item(), loss.item()))
        loss.backward(retain_graph=True)
        optimizer.step()

        if(itr%int(Nitr/10)==0):
            saveimgs(images, imgpath=os.path.join(directory, "%i.png"%itr))
            # saveimgs(images, imgpath=os.path.join(directory, "%i_loss%f.png"%(itr, loss.to('cpu').detach().numpy() )))

    # save the best value
    saveimgs(images, imgpath=os.path.join(directory, "best_%0.4f.png"%loss.to('cpu').detach().numpy() ))
    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    plt.scatter(y_list, list_loss)
    plt.savefig(os.path.join(directory, "loss.png"))

def optuna_objective(trial): ### for optuna
    x = trial.suggest_uniform('x', -0.1, 0.1)
    y = trial.suggest_uniform('y', min_y, max_y)
    z = trial.suggest_uniform('z', -0.1, 0.1)

    cfg = const_cfg()
    cams = create_cams(cfg)
    meshes, subdiv, _, _ = nvdiff_meshes(cfg)
    render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)

    images= render_images(torch.tensor([x, y, z]), render_meshes, cams, cfg) 

    score = calc_loss(images)    

    if(trial.number%int(Nitr/10)==0):
        saveimgs(images, imgpath=os.path.join(directory, "%i.png"%trial.number))

    print('iter: %i, %1.3f, %1.3f, %1.3f score: %1.3f' % (trial.number, x, y, z, score))
    return score


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

# renderer = construct_renderer(batch_size=batch_size)
# mesh_A, mesh_B = construct_scene(device=device)

#### bruteforce
if(args.bruteforce):
    directory = os.path.join("trans_images", query.replace(" ", "_"), "bruteforce")
    os.makedirs(directory, exist_ok = True)
    opt_bruteforce(Nitr, directory)

### optimize with autograd
if(args.autograd):
    print("start optimizing with autograd of pytorch")
    directory = os.path.join("trans_images", query.replace(" ", "_"), "autograd")
    os.makedirs(directory, exist_ok = True)
    opt_autograd(Nitr, directory)

#### optuna 
if(args.optuna):
    print("start optimizing with optuna")
    directory = os.path.join("trans_images", query.replace(" ", "_"), "optuna")
    os.makedirs(directory, exist_ok = True)

    study = optuna.create_study(direction="minimize") # 最適化処理を管理するstudyオブジェクト
    study.optimize(optuna_objective, # 目的関数
                n_trials=Nitr # トライアル数
                )

    cfg = const_cfg()
    cams = create_cams(cfg)
    meshes, subdiv, _, _ = nvdiff_meshes(cfg)
    render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)

    images = render_images(torch.tensor([study.best_params["x"], study.best_params["y"], study.best_params["z"]]), render_meshes, cams, cfg) 
    # image_grid(images.cpu().detach().numpy(), rows=2, cols=3, rgb=True, path=os.path.join(directory, "best_%f.png"%study.best_params["y"]))
    saveimgs(images, imgpath=os.path.join(directory, "best_%0.4f.png"%study.best_params["y"])) #"loss_%f_atitr_%i.png"%(score, itr))

    fig = plt.figure()
    plt.figure(figsize=(10, 10))
    plt.scatter([trial.params["y"] for trial in study.trials], [trial.value for trial in study.trials])
    plt.savefig(os.path.join(directory, "loss.png"))

# if (args.testimage ):
    # images = render_images(torch.tensor([0,0,0])) 
    # import ipdb; ipdb.set_trace()
    # calc_loss(images)
    # image_grid(images.cpu().detach().numpy(), rows=2, cols=3, rgb=True, path=os.path.join(directory, "best_%f.png"%study.best_params["y"]))

# optuna.visualization.matplotlib.plot_optimization_history(study)
# plt.tight_layout()
# path = os.path.join(directory, "optuna_plot.png")
# plt.savefig(path)
# print("best", study.best_value)