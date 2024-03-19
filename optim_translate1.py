import torch
import clip
import argparse

# import pytorch3d as p3d
import trimesh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from plot_image_grid import image_grid

from render_nvdiff import const_cfg, nvdiff_meshes, create_cams, nvdiff_rendermeshes, const_scene_camparams, const_trainrender, const_cfg_batch1, const_trainrender1, const_cfg1

from PIL import Image, ImageDraw, ImageFont

import os
import optuna

import torchvision

from datetime import datetime
import pytz
from mesh_overlap import calc_overlap
from center_check import calc_center
# from mesh_physical import calc_kinetic_energy
import copy

jst = pytz.timezone('Asia/Tokyo')

def save_image_with_trial_number(image, trial_number, save_path):
     # PyTorchのテンソルをNumPy配列に変換
    image_array = image.detach().cpu().numpy()

    # 画像をPillowのImageオブジェクトに変換
    img = Image.fromarray(image_array)

    # テキストを追加
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()  # フォントを選択

    # テキストの位置と内容を指定
    text = f"Trial Number: {trial_number}"
    position = (10, 10)  # テキストの左上の座標
    text_color = (0, 0, 0)  # テキストの色 (黒)

    # テキストを画像に描画
    draw.text(position, text, fill=text_color, font=font)

    # 画像を保存
    img.save(save_path)


now = datetime.now(jst)
date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

torch.set_printoptions(precision=3)

cosine_sim = torch.nn.CosineSimilarity()
device = "cuda"
max_y = 1.0
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

def render_images_optuna_test(trans, render_meshes, cams, cfg, cfg1, cams1):

    # render_meshes_original = render_meshes.copy().deepcopy()

    # add diff of trans to the meshes
    # for mesh in render_meshes:
    overlap_score = 0
    center_score = 0
    kinetic_energy_score = 0
    render_meshes[1].v_pos = render_meshes[1].v_pos.clone().detach() + trans.to(device)
    complete_scene, params_camera  = const_scene_camparams(render_meshes, cams)
    images = const_trainrender(complete_scene, params_camera, cfg)
    # overlap_score = calc_overlap(render_meshes[0], render_meshes[1])
    # center_score = calc_center(render_meshes[0], render_meshes[1])
    # kinetic_energy_score = calc_kinetic_energy(render_meshes[0], render_meshes[1])
    # complete_scene1, params_camera1  = const_scene_camparams(render_meshes, cams1)
    # image1 = const_trainrender1(complete_scene1, params_camera1, cfg1)

    # bring back mesh positions
    # for mesh in render_meshes:
    render_meshes[1].v_pos = render_meshes[1].v_pos.clone().detach() -  trans.to(device)

    return images, overlap_score, center_score, kinetic_energy_score#, image1

def calc_loss(images):
    #cosine_similarity
    image_features = model.encode_image(images)
    text_features = model.encode_text(text)
    text_features = text_features.expand(image_features.shape)
    loss = cosine_avg(image_features, text_features)

    # contrastive_loss
    # image_features = model.encode_image(images)
    # text_features = model.encode_text(text)
    # similarities = torch.matmul(image_features, text_features.T)
    # mask = torch.ones_like(similarities) - torch.eye(len(images), len(text), dtype=torch.long, device=similarities.device)
    # similarities = similarities.masked_select(mask.bool())
    # if len(similarities) % 2 != 0:
    #     similarities = similarities[:len(similarities)-1]
    # similarities = similarities.view(-1, 2)
    # labels = torch.arange(0, similarities.size(0), dtype=torch.long, device=similarities.device)
    # targets = torch.ones_like(labels)
    # loss_fn = torch.nn.MarginRankingLoss(margin=0.2)
    # loss = loss_fn(similarities[:, 0], similarities[:, 1], targets)
    
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

    xyz_list = torch.meshgrid(torch.linspace(min_y, max_y, Nitr),
                              torch.linspace(min_y, max_y, Nitr),
                              torch.linspace(min_y, max_y, Nitr))

    trans_all = torch.stack([xyz_list[0].flatten(), xyz_list[1].flatten(), xyz_list[2].flatten()], dim=1)

    list_loss = []
    loss_min = 100.0
    arg_xyz = []
    min_itr = 0

    # 最適化の結果として得られた格子点の座標と損失を収集するリスト
    xyz_coords_list = []

    cfg = const_cfg()
    cfg1 =const_cfg1()
    cams = create_cams(cfg)
    cams1 = create_cams(cfg1)
    meshes, subdiv, _, _ = nvdiff_meshes(cfg)
    render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)

    for itr, trans in enumerate(trans_all):
        # render_meshes
        images, overlap_score, center_score, _ = render_images_optuna_test(trans, render_meshes, cams, cfg, cfg1, cams1) 
        if(itr%int(100)==0):
            saveimgs(images, imgpath=os.path.join(directory, "%i.png"%itr))
        loss = calc_loss(images).to('cpu').detach().numpy() - overlap_score - center_score
        list_loss.append(loss)
        xyz_coords_list.append(trans)
        # import ipdb; ipdb.set_trace()
        if(loss < loss_min):
            loss_min = loss
            print("updated at: ", itr, trans, loss)
            arg_xyz = trans
            min_itr = itr

    print("min itr", min_itr, "min trans: ", arg_xyz, "loss: ", loss_min)

    result_file_path = os.path.join(directory, "best_params.txt")
    with open(result_file_path, 'w') as result_file:
        result_file.write(f"Best trial number: {min_itr}\n")
        result_file.write(f"Best x: {arg_xyz[0]}\n")
        result_file.write(f"Best y: {arg_xyz[1]}\n")
        result_file.write(f"Best z: {arg_xyz[2]}\n")

    images= render_images(arg_xyz, render_meshes, cams, cfg) 
 
    saveimgs(images, imgpath=os.path.join(directory, "opt_min_at_%i.png"%min_itr))

    # カラーマップを逆順にして、損失が小さいほど赤色にする
    cmap = plt.cm.get_cmap('RdYlBu')

    fig = plt.figure(figsize=(10, 10))
    print("fig = plt.figure()")
    ax = fig.add_subplot(111, projection="3d")
    # print(xyz_coords_list[0][1])
    # print(list_loss[0])

    # 損失の最小値と最大値を取得
    min_loss = min(list_loss)
    max_loss = max(list_loss)
    # min_loss = -0.30
    # max_loss = -0.20

    # 正規化オブジェクトを作成
    norm = plt.Normalize(vmin=min_loss, vmax=max_loss)

    # 各イテレーションでの座標と損失をプロット
    for i in range(len(xyz_coords_list)):
        scatter = ax.scatter(xyz_coords_list[i][0], xyz_coords_list[i][1], xyz_coords_list[i][2], 
                            c=[list_loss[i]], cmap=cmap, norm=norm)

    # カラーバーを追加
    plt.colorbar(scatter, label='Loss')

    # 軸ラベルを設定
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.savefig(os.path.join(directory, "loss.png"))

    # 最適なy変換値を取得
    min_loss_index = list_loss.index(min(list_loss))
    best_xyz = trans_all[min_loss_index].tolist()  # 最適なxyz変換値

    # best_yを使ってオブジェクトを配置してレンダリング
    cfg = const_cfg_batch1()
    cams = create_cams(cfg)
    meshes, subdiv, _, _ = nvdiff_meshes(cfg)
    render_meshes, _ = nvdiff_rendermeshes(meshes, subdiv, cfg)

    images = render_images(torch.tensor([best_xyz[0], best_xyz[1], best_xyz[2]]), render_meshes, cams, cfg)

    # 画像をファイルに保存
    # saveimgs(images, imgpath=os.path.join(directory, "rendered_with_best_xyz.png"))

    np.savetxt(os.path.join(directory,'list_loss.txt'), list_loss)
    # xyz_coords_list内の各要素を1つのテンソルに平坦化
    reshaped_coords_flat = [torch.cat([coords]).flatten() for coords in xyz_coords_list]

    # ファイルに保存
    np.savetxt(os.path.join(directory,'xyz_coords_list.txt'), [coords.numpy() for coords in reshaped_coords_flat])


def save_optimal_obj(best_y, mesh_path, output_path):
    # Load the original mesh
    m = trimesh.load(mesh_path)

    # Clone the original vertices and add the optimal y value
    vertices = m.vertices.copy()
    vertices[:, 1] += best_y

    # Create a new Trimesh object with the adjusted vertices
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=m.faces)

    # Export the new mesh to an obj file
    new_mesh.export(output_path)

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

best_scores = {}  # 各区間の最高スコアを保存する辞書

def optuna_objective(trial): ### for optuna
    # if trial.number == 0:
    #     x =  trial.suggest_uniform('x', 2, 2)
    #     y =  trial.suggest_uniform('y', 2, 2)
    #     z =  trial.suggest_uniform('z', 2, 2)
    # else:
    #     x = trial.suggest_uniform('x', min_y, max_y)
    #     y = trial.suggest_uniform('y', min_y, max_y)
    #     z = trial.suggest_uniform('z', min_y, max_y)
    
    x = trial.suggest_uniform('x', min_y, max_y)
    y = trial.suggest_uniform('y', min_y, max_y)
    z = trial.suggest_uniform('z', min_y, max_y)

    cfg = const_cfg()
    cfg1 =const_cfg1()
    cams = create_cams(cfg)
    cams1 = create_cams(cfg1)
    meshes, subdiv, _, _ = nvdiff_meshes(cfg)
    render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)

    images, overlap_score, center_score, kinetic_energy_score, image1= render_images_optuna_test(torch.tensor([x, y, z]), render_meshes, cams, cfg, cfg1, cams1) 

    score = calc_loss(images) #- overlap_score - center_score #- kinetic_energy_score  

    # if(trial.number%int(Nitr/10)==0):
    #     saveimgs(images, imgpath=os.path.join(directory, "%i.png"%trial.number))

    saveimgs(image1, imgpath=os.path.join(directory, "%i.png"%trial.number))

      # トライアル番号とスコアをテキストファイルに書き込む
    # with open("all_scores.txt", "a") as f:
    #     f.write(f"Iteration {trial.number}: Trial Number - {trial.number}, Score - {score}\n")

    # 10区間ごとに最高スコアを保存　メモリ食いすぎる
    # group_index = trial.number // 10
    # if group_index not in best_scores:
    #     best_scores[group_index] = (trial.number, score)
    # else:
    #     best_trial_number, best_score = best_scores[group_index]
    #     if score > best_score:
    #         best_scores[group_index] = (trial.number, score)

    # if group_index - 10 in best_scores:
    #     del best_scores[group_index - 10]


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

#not good result
# model.load_state_dict(torch.load("lr_1e-6.pth"))

#test
model.load_state_dict(torch.load("/home/itoh/ClipMesh/finetuning_models/all_3_d_p_1e-6/all_3_d_p.pth"))

query = "A photo of stable chair"
# query = "An image of black silk hat on top of a head"
# query = "An image of head and black hat far away separated"
# query = "an image of two separated objects on white background"

text = clip.tokenize([query]).to(device)

# set render, meshes
# Nitr = 200
Nitr = 10
batch_size = 12

# renderer = construct_renderer(batch_size=batch_size)
# mesh_A, mesh_B = construct_scene(device=device)

#### bruteforce
if(args.bruteforce):
    directory = os.path.join(f"trans_images_{date_time}", query.replace(" ", "_"), "bruteforce")
    os.makedirs(directory, exist_ok = True)
    opt_bruteforce(Nitr, directory)

### optimize with autograd
if(args.autograd):
    print("start optimizing with autograd of pytorch")
    directory = os.path.join(f"trans_images_{date_time}", query.replace(" ", "_"), "autograd")
    os.makedirs(directory, exist_ok = True)
    opt_autograd(Nitr, directory)

#### optuna 
if(args.optuna):
    print("start optimizing with optuna")
    directory = os.path.join(f"trans_images_{date_time}", query.replace(" ", "_"), "optuna")
    os.makedirs(directory, exist_ok = True)

    study = optuna.create_study(direction="minimize") # 最適化処理を管理するstudyオブジェクト
    study.optimize(optuna_objective, # 目的関数
                n_trials=Nitr # トライアル数
                )

    cfg = const_cfg()
    cams = create_cams(cfg)
    meshes, subdiv, _, _ = nvdiff_meshes(cfg)
    render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)

    # images = render_images(torch.tensor([study.best_params["y"]]), render_meshes, cams, cfg)
    images = render_images(torch.tensor([study.best_params["x"], study.best_params["y"], study.best_params["z"]]), render_meshes, cams, cfg) 
    # image_grid(images.cpu().detach().numpy(), rows=2, cols=3, rgb=True, path=os.path.join(directory, "best_%f.png"%study.best_params["y"]))
    # 最も低い損失を持つトライアルを見つける
    best_trial = min(study.trials, key=lambda trial: trial.value)
    # トライアル番号をファイル名として使用して画像を保存
    saveimgs(images, imgpath=os.path.join(directory, f"best_trial_{best_trial.number}_loss_{best_trial.value:.4f}.png"))

    # 最適なパラメータを取得
    best_x = study.best_params["x"]
    best_y = study.best_params["y"]
    best_z = study.best_params["z"]
    best_trial_number = best_trial.number

     # テキストファイルに最適なパラメータを保存
    result_file_path = os.path.join(directory, "best_params.txt")
    with open(result_file_path, 'w') as result_file:
        result_file.write(f"Best trial number: {best_trial_number}\n")
        result_file.write(f"Best x: {best_x}\n")
        result_file.write(f"Best y: {best_y}\n")
        result_file.write(f"Best z: {best_z}\n")


    # プロットを作成
    fig, ax = plt.subplots(figsize=(10, 10))
    x = range(len(study.trials))
    y = [trial.value for trial in study.trials]
    ax.scatter(x, y)

    # 横軸ラベルを追加
    ax.set_xlabel('iteration')

    # 縦軸ラベルを追加
    ax.set_ylabel('score')

    # グラフのタイトルを追加
    ax.set_title('Loss')

    plt.savefig(os.path.join(directory, "loss.png"))

# 一つのファイルに全ての最適なスコアを保存
# filename = "best_scores.txt"
# with open(filename, "w") as f:
#     for group, (best_trial_number, best_score) in best_scores.items():
#         f.write(f"Group {group}: Best Trial Number - {best_trial_number}, Best Score - {best_score}\n")

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