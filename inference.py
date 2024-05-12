import net
import torch
import os
from face_alignment import align
import numpy as np


import warnings
warnings.filterwarnings("ignore")
from PIL import Image

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')  # 将图像转换为RGB模式
    image = image.resize((112, 112)) 
    return image


adaface_models = {
    'ir_50': "pretrained/adaface_ir50_ms1mv2.ckpt",
}


def load_pretrained_model(architecture='ir_50'):
    # load model and pretrained statedict
    assert architecture in adaface_models.keys()
    model = net.build_model(architecture)
    statedict = torch.load(adaface_models[architecture])['state_dict']
    model_statedict = {key[6:]: val for key, val in statedict.items() if key.startswith('model.')}
    model.load_state_dict(model_statedict)
    model.eval()
    return model


def to_input(pil_rgb_image):
    np_img = np.array(pil_rgb_image)
    brg_img = ((np_img[:, :, ::-1] / 255.) - 0.5) / 0.5
    tensor = torch.tensor([brg_img.transpose(2, 0, 1)]).float()
    return tensor


if __name__ == '__main__':
    import csv
    model = load_pretrained_model('ir_50')
    # 读取第一张图片并提取特征
    image1 = load_image('./SR/021564.png')
    feature1, norm = model(to_input(image1))

    # 读取第二张图片并提取特征
    image2 = load_image('./HR/021564.png')
    feature2, norm = model(to_input(image2))

    similarity = torch.nn.functional.cosine_similarity(feature1, feature2, dim=1)
    print(similarity)
# 遍历文件夹中的所有图片
sr_folder = './SR'
hr_folder = './HR'

# 获取文件夹中的所有文件名
sr_files = os.listdir(sr_folder)
hr_files = os.listdir(hr_folder)

# 打开CSV文件以写入结果
csv_file = open('similarity_results.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['Filename', 'Similarity'])

# 循环处理每个文件
for filename in sr_files:
    # 检查是否有对应的HR文件
    if filename in hr_files:
        # 读取SR图片并提取特征
        sr_image = load_image(os.path.join(sr_folder, filename))
        sr_feature, _ = model(to_input(sr_image))

        # 读取对应的HR图片并提取特征
        hr_image = load_image(os.path.join(hr_folder, filename))
        hr_feature, _ = model(to_input(hr_image))

        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(sr_feature, hr_feature, dim=1)
        similarity_value = similarity.item()

        # 将结果写入CSV文件
        csv_writer.writerow([filename, similarity_value])
        print(f"Filename: {filename}, Similarity: {similarity_value}")

    else:
        print(f"No corresponding HR image found for {filename}")

# 关闭CSV文件
csv_file.close()

