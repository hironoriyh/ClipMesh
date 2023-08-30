import trimesh
import numpy as np

# OBJファイルのパスを指定
obj_file_path = "/home/itoh/ClipMesh/3DCoMPaT-v2/loaders/3D/p_chair_separate_f_legs.obj"

# OBJファイルを読み込む
mesh = trimesh.load_mesh(obj_file_path)

output_path = 'test_p_chair.obj'

# y軸周りに90度回転するための回転行列
rotation_angle_y = 90  # 度数法で指定（例えば90度回転）
rotation_matrix_y = trimesh.transformations.rotation_matrix(np.radians(rotation_angle_y), [0, 1, 0])


# メッシュをy軸周りに90度回転させる
mesh.apply_transform(rotation_matrix_y)

mesh.export(output_path, file_type='obj')

