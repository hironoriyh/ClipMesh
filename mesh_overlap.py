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

cfg = const_cfg()
cams = create_cams(cfg)
meshes, subdiv, _, _ = nvdiff_meshes(cfg)
render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)

# print(dir(render_meshes[0]))

print(type(render_meshes[0]))

# メッシュ1の頂点座標と面のインデックス
vertices1 = render_meshes[0].v_pos.detach().cpu().numpy()  # メッシュ1の頂点座標
faces1 = render_meshes[0].t_pos_idx.detach().cpu().numpy()   # メッシュ1の面のインデックス

# メッシュ2の頂点座標と面のインデックス
vertices2 = render_meshes[1].v_pos.detach().cpu().numpy()  # メッシュ2の頂点座標
faces2 = render_meshes[1].t_pos_idx.detach().cpu().numpy()   # メッシュ2の面のインデックス


mesh1 = trimesh.Trimesh(vertices=vertices1, faces=faces1)
mesh2 = trimesh.Trimesh(vertices=vertices2, faces=faces2)

#椅子上部の最低点
bottom_point = mesh2.vertices[:, 1].min()
#椅子脚部の最高点
top_point = mesh1.vertices[:, 1].max()

# top_point - bottom_point の値を計算
value = top_point - bottom_point

# スコアを計算する。ここでは逆数を取り、0に近づくほどスコアが上昇するようにする。
#スコアは正の値なのでマイナスをつける必要がある
score = 0.2 / (abs(value) + 1)

def calc_overlap(mesh1, mesh2):
    # メッシュ1の頂点座標と面のインデックス
    vertices1 = mesh1.v_pos.detach().cpu().numpy()  # メッシュ1の頂点座標
    faces1 = mesh1.t_pos_idx.detach().cpu().numpy()   # メッシュ1の面のインデックス

    # メッシュ2の頂点座標と面のインデックス
    vertices2 = mesh2.v_pos.detach().cpu().numpy()  # メッシュ2の頂点座標
    faces2 = mesh2.t_pos_idx.detach().cpu().numpy()   # メッシュ2の面のインデックス


    mesh1 = trimesh.Trimesh(vertices=vertices1, faces=faces1)
    mesh2 = trimesh.Trimesh(vertices=vertices2, faces=faces2)

    #椅子上部の最低点
    bottom_point = mesh2.vertices[:, 1].min()
    #椅子脚部の最高点
    top_point = mesh1.vertices[:, 1].max()

    # top_point - bottom_point の値を計算
    value = top_point - bottom_point

    # スコアを計算する。ここでは逆数を取り、0に近づくほどスコアが上昇するようにする。
    #スコアは正の値なのでマイナスをつける必要がある
    overlap_score = 0.07 / (abs(value) + 1)

    return overlap_score

# # メッシュ同士の位置関係を判定
# if top_point > bottom_point:
#     print("くっついている")
# else:
#     print("離れている")










