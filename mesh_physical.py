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

import pybullet as p
import pybullet_data
import time

cfg = const_cfg()
cams = create_cams(cfg)
meshes, subdiv, _, _ = nvdiff_meshes(cfg)
render_meshes, _ =  nvdiff_rendermeshes(meshes, subdiv, cfg)

# メッシュ1の頂点座標と面のインデックス
vertices1 = render_meshes[0].v_pos.detach().cpu().numpy()  # メッシュ1の頂点座標
faces1 = render_meshes[0].t_pos_idx.detach().cpu().numpy()   # メッシュ1の面のインデックス

# メッシュ2の頂点座標と面のインデックス
vertices2 = render_meshes[1].v_pos.detach().cpu().numpy()  # メッシュ2の頂点座標
faces2 = render_meshes[1].t_pos_idx.detach().cpu().numpy()   # メッシュ2の面のインデックス

def scale_value(value, min_value, max_value, scaled_min, scaled_max):
    # 指定した範囲にスケールする
    scaled_value = (value - min_value) / (max_value - min_value) * (scaled_max - scaled_min) + scaled_min
    
    # スケールされた値を範囲内に制限
    scaled_value = max(scaled_min, min(scaled_max, scaled_value))
    
    return scaled_value

def calc_kinetic_energy(mesh1, mesh2):
    min_value = 0  # 値の最小値
    max_value = 200  # 値の最大値
    scaled_min = 0  # 正規化後の最小値
    scaled_max = 0.07  # 正規化後の最大値

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
    value = bottom_point - top_point

    # 初期化
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    # planeId = p.loadURDF("plane.urdf", basePosition=[0, 0, -1.12]) #-2.25

    if value <  -0.1:
        # p.disconnect()
        allKineticEnergy = 400
        print("椅子の上部が脚よりも下に位置しています")
        scaled_value = scale_value(allKineticEnergy, min_value, max_value, scaled_min, scaled_max)
        #マイナスの値にする
        kinetic_loss = scaled_max - scaled_value
        return kinetic_loss  # すぐに400を返す


    # 物体のロード
    center1 = np.mean(mesh1.vertices, axis=0)
    center2= np.mean(mesh2.vertices, axis=0)
    # y_coordinates = mesh2.vertices[:, 1]
    # min_y_value = np.min(y_coordinates)

    # print("Y座標の最小値:", min_y_value)
    # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    # print("center1:", center1)
    # print("center2:", center2)

    startPos1 = [center1[2], center1[0], center1[1]]
    # startPos1 = [0, 0, -0.8]
    # print("pos1:", startPos1)
    startOrientation1 = p.getQuaternionFromEuler([1.5, 0, 0])
    boxId1 = p.loadURDF("/home/itoh/ClipMesh/scene21_leg.urdf", startPos1, startOrientation1, globalScaling=2.859659950805005, useFixedBase=True)
    # boxId1 = p.loadURDF("/home/itoh/ClipMesh/chair_no_seat.urdf", startPos1, startOrientation1)

    startPos2 = [center2[2], center2[0], center2[1]]
    # startPos2 = [0, 0, -1]
    # print("pos2:", startPos2)
    startOrientation2 = p.getQuaternionFromEuler([1.5, 0, 0])
    boxId2 = p.loadURDF("/home/itoh/ClipMesh/p_chair_up.urdf", startPos2, startOrientation2, globalScaling=0.6122264466450626)
    # boxId2 = p.loadURDF("/home/itoh/ClipMesh/chair_no_lega.urdf", startPos2, startOrientation2)

    # シミュレーションステップ数
    max_steps = 150
    current_step = 0


    # 運動エネルギーの変数を初期化
    totalKineticEnergy1 = 0
    totalKineticEnergy2 = 0
    collision_step_count = max_steps
    collision_detected = False

    # startPos1とstartPos2が1以上離れているかどうかを確認
    if np.linalg.norm(np.array(startPos1) - np.array(startPos2)) >= 1.5:
        allKineticEnergy = 400
        print("大きく離れています")
        scaled_value = scale_value(allKineticEnergy, min_value, max_value, scaled_min, scaled_max)
        #マイナスの値にする
        kinetic_loss = scaled_max - scaled_value
        return kinetic_loss  # すぐに400を返す

    while current_step < max_steps:
        p.stepSimulation()
        time.sleep(1.0 / 240.0)
        current_step += 1
        # print(current_step)

        # 接触情報の取得
        contacts = p.getContactPoints(bodyA=boxId1, bodyB=boxId2)

        if contacts:
            # 衝突している場合、collision_step_countを30に設定し、
            # 固定ジョイントを作成
            if not collision_detected:
                collision_detected = True
                collision_step_count = 30
                totalKineticEnergy2 = 0
                # contact_point = contacts[0]  # 最初の接触点を使用
                # pivot_point = contact_point[5]  # 接触点の座標
                # p.createConstraint(boxId1, -1, boxId2, -1, p.JOINT_FIXED, pivot_point, [0, 0, 0], [0, 0, 0])

        if collision_detected:
            # 運動エネルギーの取得（boxId2）
            baseVelocity2 = p.getBaseVelocity(boxId2)
            linearVelocity2 = baseVelocity2[0]
            angularVelocity2 = baseVelocity2[1]

            # 運動エネルギーを計算（boxId1）
            linearKineticEnergy2 = 0.5 * (
                linearVelocity2[0] ** 2 + linearVelocity2[1] ** 2 + linearVelocity2[2] ** 2
            )
            angularKineticEnergy2 = 0.5 * (
                angularVelocity2[0] ** 2 + angularVelocity2[1] ** 2 + angularVelocity2[2] ** 2
            )

            # 運動エネルギーを累積
            totalKineticEnergy2 += linearKineticEnergy2 + angularKineticEnergy2
            # print (totalKineticEnergy1)

            # collision_step_countが0になったらシミュレーションを終了
            if collision_step_count == 0:
                break

            # collision_step_countを1減少
            collision_step_count -= 1

        # 運動エネルギーを表示
    if collision_detected:
        allKineticEnergy = totalKineticEnergy2
        # print("運動エネルギー1:", totalKineticEnergy1)
        # print("運動エネルギー2:", totalKineticEnergy2)
        print("上部の運動エネルギー:", allKineticEnergy)

    else:
        allKineticEnergy = 400

        print("接触しませんでした:", allKineticEnergy)

    # print("pos1:", startPos1)
    # print("pos2:", startPos2)

    scaled_value = scale_value(allKineticEnergy, min_value, max_value, scaled_min, scaled_max)
    #マイナスの値にする
    kinetic_loss = scaled_max - scaled_value
    p.disconnect()
    return kinetic_loss

calc_kinetic_energy(render_meshes[0], render_meshes[1])










