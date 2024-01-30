import torch
import clip
import argparse

# import pytorch3d as p3d
import trimesh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from plot_image_grid import image_grid

from render_nvdiff import const_cfg, nvdiff_meshes, create_cams, nvdiff_rendermeshes, const_scene_camparams, const_trainrender, const_cfg_batch1

from PIL import Image
import os
import optuna

import torchvision
import pymesh
import numpy as np
import math

device = "cuda"

cfg = const_cfg()
cams = create_cams(cfg)
meshes, subdiv, _, _ = nvdiff_meshes(cfg)
render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)

trans10_vanilla1 = [
    [0, 1.4230424777173065, 0],
    [0, 1.2860969722497901, 0],
    [0, 1.3523870873184305, 0],
    [0, 1.5393792230229812, 0],
    [0, 1.2014711316869253, 0],
    [0, 1.190315535391053, 0],
    [0, 0.5622911232500721, 0],
    [0, 1.7332672858448421, 0],
    [0, 1.1253425875168732, 0],
    [0, 0.2877444099271263, 0]
]

trans10_vanilla3 = [
    [ 0.05370556515268854, 1.3629764374447324, 1.160083831670283],
    [-1.7918797692202892, 0.7755426892953683, -0.37748828324945627],
    [-1.4523800174182826, 1.4807049106487429, -0.597433271094568],
    [0.4421634350947782, 1.2398892954837903, 0.1419269584909024],
    [-2.57605159495275, 0.7999380326598908, 0.47974994720824826],
    [0.32503697144235955, 0.9282939737540714, -0.016245629141585116],
    [-0.6983129124747378, 1.2524189339257419, -0.8935595603399316],
    [0.11833410674434958, 1.2077255922177326, -0.2071761584462923],
    [0.04326376124639719, 1.5562970432667695, -0.8133791882212748],
    [-2.2418103606238535, 0.8575806909380146, -0.027960068751561207]
]

trans10_vanilla_kikagaku = [
    [0.13642634689085043, 0.8060862315724304, 0.04524335480421207],
    [0.07080053528284913, 0.24229149308130982, -0.01054084796712329],
    [-0.17384864678029996, 0.8312411093610959, -0.24890680139258153],
    [0.11370665984684286, 0.8066634024041348, 0.051921945028375105],
    [0.020122498002634104, 0.7079151086518508, 0.39621972205848327],
    [-0.02443001465514405, 0.5000458895815102, -0.023208897140772844],
    [-0.15017543603237948, 0.8797404555212511, -0.024401385158008992],
    [0.15762233349838456, 0.9101243098971099, 0.12135197400106051],
    [-0.12698556504834752, 0.6781531582859802, 0.14445311398651878],
    [0.17726412670185404, 0.8674687125449966, 0.014722687791135225]
]

trans10_fine_tuning = [
    [-0.22291731738499299, 0.7842152623358636, 0.11933224789153474],
    [-0.7513439207762913, 0.7792711048398223, 0.1277080798411],
    [-0.08758450889311209, 0.7841575953689901, 0.23330228287361954],
    [0.39865588356898424, 0.8380451446087543, -0.3067008608636159],
    [-0.3944986382321882, 0.5273629417531804, -0.19746404040850174],
    [-0.3452380555142309, 0.09255658791250776, -0.099297182981395],
    [-1.2923242585417265, 0.5204771695921416, 0.109130640210851],
    [-1.454872503895682, 0.9834677805353635, 0.2261189106782167],
    [-0.30631389292938777, 0.4461178939631227, 0.19524416536770034],
    [-0.03549558465668712, 1.1776799239636992, 0.03596673989066401]
]


# ファイルへの書き込み用のテキストファイルをオープン
with open("distances_l1.txt", "w") as file:
    for trans_values in trans10_fine_tuning :
        trans = torch.Tensor(trans_values)
        # print(render_meshes[1].v_pos)
        render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)
        render_meshes[1].v_pos = render_meshes[1].v_pos.clone().detach() + trans.to(device)
        # print(render_meshes[1].v_pos)

        vertices1 = render_meshes[0].v_pos.detach().cpu().numpy()
        faces1 = render_meshes[0].t_pos_idx.detach().cpu().numpy()
        vertices2 = render_meshes[1].v_pos.detach().cpu().numpy()
        faces2 = render_meshes[1].t_pos_idx.detach().cpu().numpy()

        # print(vertices2)

        mesh1 = trimesh.Trimesh(vertices=vertices1, faces=faces1)
        mesh2 = trimesh.Trimesh(vertices=vertices2, faces=faces2)

        scene1 = trimesh.scene.Scene()
        scene1.add_geometry(mesh1)

        scene2 = trimesh.scene.Scene()
        scene2.add_geometry(mesh2)

        scene1_center = scene1.centroid
        scene2_center = scene2.centroid
        # print(scene1_center)
        # print(scene2_center)

        distance = math.sqrt((scene1_center[0] - scene2_center[0])**2 + (scene1_center[2] - scene2_center[2])**2)
        print(f"Distance for trans {trans_values}: {distance}")

        # ファイルに距離を書き込む
        file.write(f"Distance for trans {trans_values}: {distance}\n")

def calc_center(mesh1,mesh2):
    
    vertices1 = mesh1.v_pos.detach().cpu().numpy()  # メッシュ1の頂点座標
    faces1 = mesh1.t_pos_idx.detach().cpu().numpy()   # メッシュ1の面のインデックス

    # メッシュ2の頂点座標と面のインデックス
    vertices2 = mesh2.v_pos.detach().cpu().numpy()  # メッシュ2の頂点座標
    faces2 = mesh2.t_pos_idx.detach().cpu().numpy()   # メッシュ2の面のインデックス


    mesh1 = trimesh.Trimesh(vertices=vertices1, faces=faces1)
    mesh2 = trimesh.Trimesh(vertices=vertices2, faces=faces2)
    
    # trimesh.sceneオブジェクトを作成します
    scene1 = trimesh.scene.Scene()

    # mesh1をsceneに追加します
    scene1.add_geometry(mesh1)

    # trimesh.sceneオブジェクトを作成します
    scene2 = trimesh.scene.Scene()

    # mesh1をsceneに追加します
    scene2.add_geometry(mesh2)

    scene1_center = scene1.centroid
    scene2_center = scene2.centroid

    # print(scene1_center)
    # print(scene2_center)
    distance = math.sqrt((scene1_center[0] - scene2_center[0])**2 + (scene1_center[2] - scene2_center[2])**2)
    center_score = 0.06/ (abs(distance) + 1)
    # print("center_score:", center_score)

    return center_score















