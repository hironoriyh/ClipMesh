import os
import json

# メインの画像フォルダのパス
stable_image_folder = "/home/itoh/ClipMesh/3DCoMPaT-v2/loaders/3D/scene_all_stable"
unstable_image_detach = "/home/itoh/ClipMesh/3DCoMPaT-v2/loaders/3D/scene_all_unstable_detach"
unstable_image_penetrate = "/home/itoh/ClipMesh/3DCoMPaT-v2/loaders/3D/scene_all_unstable_penetrate"

# データを格納するリストを初期化
all_images_data = []

unstable_caption_idx = 0
stable_caption_idx = 0

openai_imagenet_template = [
    lambda c: f'a bad photo of a {c}.',
    lambda c: f'a photo of many {c}.',
    lambda c: f'a sculpture of a {c}.',
    lambda c: f'a photo of the hard to see {c}.',
    lambda c: f'a low resolution photo of the {c}.',
    lambda c: f'a rendering of a {c}.',
    lambda c: f'graffiti of a {c}.',
    lambda c: f'a bad photo of the {c}.',
    lambda c: f'a cropped photo of the {c}.',
    lambda c: f'a tattoo of a {c}.',
    lambda c: f'the embroidered {c}.',
    lambda c: f'a photo of a hard to see {c}.',
    lambda c: f'a bright photo of a {c}.',
    lambda c: f'a photo of a clean {c}.',
    lambda c: f'a photo of a dirty {c}.',
    lambda c: f'a dark photo of the {c}.',
    lambda c: f'a drawing of a {c}.',
    lambda c: f'a photo of my {c}.',
    lambda c: f'the plastic {c}.',
    lambda c: f'a photo of the cool {c}.',
    lambda c: f'a close-up photo of a {c}.',
    lambda c: f'a black and white photo of the {c}.',
    lambda c: f'a painting of the {c}.',
    lambda c: f'a painting of a {c}.',
    lambda c: f'a pixelated photo of the {c}.',
    lambda c: f'a sculpture of the {c}.',
    lambda c: f'a bright photo of the {c}.',
    lambda c: f'a cropped photo of a {c}.',
    lambda c: f'a plastic {c}.',
    lambda c: f'a photo of the dirty {c}.',
    lambda c: f'a jpeg corrupted photo of a {c}.',
    lambda c: f'a blurry photo of the {c}.',
    lambda c: f'a photo of the {c}.',
    lambda c: f'a good photo of the {c}.',
    lambda c: f'a rendering of the {c}.',
    lambda c: f'a {c} in a video game.',
    lambda c: f'a photo of one {c}.',
    lambda c: f'a doodle of a {c}.',
    lambda c: f'a close-up photo of the {c}.',
    lambda c: f'a photo of a {c}.',
    lambda c: f'the origami {c}.',
    lambda c: f'the {c} in a video game.',
    lambda c: f'a sketch of a {c}.',
    lambda c: f'a doodle of the {c}.',
    lambda c: f'a origami {c}.',
    lambda c: f'a low resolution photo of a {c}.',
    lambda c: f'the toy {c}.',
    lambda c: f'a rendition of the {c}.',
    lambda c: f'a photo of the clean {c}.',
    lambda c: f'a photo of a large {c}.',
    lambda c: f'a rendition of a {c}.',
    lambda c: f'a photo of a nice {c}.',
    lambda c: f'a photo of a weird {c}.',
    lambda c: f'a blurry photo of a {c}.',
    lambda c: f'a cartoon {c}.',
    lambda c: f'art of a {c}.',
    lambda c: f'a sketch of the {c}.',
    lambda c: f'a embroidered {c}.',
    lambda c: f'a pixelated photo of a {c}.',
    lambda c: f'itap of the {c}.',
    lambda c: f'a jpeg corrupted photo of the {c}.',
    lambda c: f'a good photo of a {c}.',
    lambda c: f'a plushie {c}.',
    lambda c: f'a photo of the nice {c}.',
    lambda c: f'a photo of the small {c}.',
    lambda c: f'a photo of the weird {c}.',
    lambda c: f'the cartoon {c}.',
    lambda c: f'art of the {c}.',
    lambda c: f'a drawing of the {c}.',
    lambda c: f'a photo of the large {c}.',
    lambda c: f'a black and white photo of a {c}.',
    lambda c: f'the plushie {c}.',
    lambda c: f'a dark photo of a {c}.',
    lambda c: f'itap of a {c}.',
    lambda c: f'graffiti of the {c}.',
    lambda c: f'a toy {c}.',
    lambda c: f'itap of my {c}.',
    lambda c: f'a photo of a cool {c}.',
    lambda c: f'a photo of a small {c}.',
    lambda c: f'a tattoo of the {c}.',
]

# 画像ファイルごとにデータを作成
for image_file in sorted(os.listdir(unstable_image_detach), key=lambda x: int(os.path.splitext(x)[0])):
    # 各画像ファイルに対してループ

    image_path = os.path.join(unstable_image_detach, image_file)
    # 画像のフルパスを取得

    unstable_caption_idx += 1
    # 不安定なキャプションのインデックスを1増やす

    # 画像に対してテンプレート内の全てのラムダ関数を使用してキャプションを生成
    for template_func in openai_imagenet_template:
        generated_caption = template_func("unstable detach chair")

        # データを辞書として追加
        all_images_data.append({"image_path": image_path, "caption": generated_caption})

# 画像ファイルごとにデータを作成
for image_file in sorted(os.listdir(unstable_image_penetrate), key=lambda x: int(os.path.splitext(x)[0])):
    # 各画像ファイルに対してループ

    image_path = os.path.join(unstable_image_penetrate, image_file)
    # 画像のフルパスを取得

    unstable_caption_idx += 1
    # 不安定なキャプションのインデックスを1増やす

    # 画像に対してテンプレート内の全てのラムダ関数を使用してキャプションを生成
    for template_func in openai_imagenet_template:
        generated_caption = template_func("unstable penetrate chair")

        # データを辞書として追加
        all_images_data.append({"image_path": image_path, "caption": generated_caption})

for image_file in sorted(os.listdir(stable_image_folder), key=lambda x: int(os.path.splitext(x)[0])):
    # 各画像ファイルに対してループ

    image_path = os.path.join(stable_image_folder, image_file)
    # 画像のフルパスを取得

    unstable_caption_idx += 1
    # 不安定なキャプションのインデックスを1増やす

    # 画像に対してテンプレート内の全てのラムダ関数を使用してキャプションを生成
    for template_func in openai_imagenet_template:
        generated_caption = template_func("stable chair")

        # データを辞書として追加
        all_images_data.append({"image_path": image_path, "caption": generated_caption})
    
# ファイルに書き込むデータを作成
all_json_data = {"images_data": all_images_data}

# JSONファイルに書き込む
with open('images_template_d_p.json', 'w') as json_file:
    json.dump(all_json_data, json_file, indent=2)

print("JSONファイルが作成されました。")