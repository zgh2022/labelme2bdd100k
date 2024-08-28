import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil
from pathlib import Path


def get_bounding_box(points):
    # 确保至少有3个点
    if len(points) < 3:
        # 如果点少于3个，返回一个简单的边界框
        x_coords, y_coords = zip(*points)
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    # 将点列表转换为numpy数组，并确保是float32类型
    points = np.array(points, dtype=np.float32)

    # 确保点集是2D的
    if len(points.shape) == 3:
        points = points.reshape(-1, 2)

    # 计算最小面积矩形
    rect = cv2.minAreaRect(points)

    # 获取矩形的四个顶点
    box = cv2.boxPoints(rect)
    box = np.int32(box)

    # 计算边界框
    x1 = int(min(box[:, 0]))
    y1 = int(min(box[:, 1]))
    x2 = int(max(box[:, 0]))
    y2 = int(max(box[:, 1]))

    return x1, y1, x2, y2


def create_bdd100k_dirs(output_dir):
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', '100k', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', '100k', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'bdd_seg_gt', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'bdd_lane_gt', split), exist_ok=True)


def process_file(json_file, json_dir, img_dir, output_dir, split):
    with open(os.path.join(json_dir, json_file), 'r') as f:
        data = json.load(f)

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    image_path = os.path.join(output_dir, 'images', '100k', split, json_file.replace('.json', '.jpg'))
    mask_path = os.path.join(output_dir, 'bdd_seg_gt', split, json_file.replace('.json', '.png'))
    lane_path = os.path.join(output_dir, 'bdd_lane_gt', split, json_file.replace('.json', '.png'))

    # 创建BDD格式的标签
    bdd_label = {
        "name": json_file.replace('.json', '.jpg'),
        "frames": [{
            "objects": []
        }]
    }

    # 处理目标检测标签 (car)
    for shape in data['shapes']:
        if shape['label'] == 'car':
            points = shape['points']
            # 确保points是一个列表的列表
            if not isinstance(points[0], list):
                points = [points]  # 如果是单个点，将其转换为列表的列表
            x1, y1, x2, y2 = get_bounding_box(points)

            bdd_label["frames"][0]["objects"].append({
                "category": "car",
                "box2d": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                }
            })

    # 创建mask图像（road分割）
    mask = np.zeros((img_height, img_width), dtype=np.uint8)
    for shape in data['shapes']:
        if shape['label'] == 'road':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

    # 创建lane图像（车道线分割）
    lane = np.zeros((img_height, img_width), dtype=np.uint8)
    for shape in data['shapes']:
        if shape['label'] == 'lane':
            points = np.array(shape['points'], dtype=np.int32)
            cv2.fillPoly(lane, [points], 255)

    # 保存图像和标签
    cv2.imwrite(mask_path, mask)
    cv2.imwrite(lane_path, lane)
    shutil.copy(os.path.join(img_dir, json_file.replace('.json', '.jpg')), image_path)

    # 保存BDD格式的JSON标签
    label_path = os.path.join(output_dir, 'labels', '100k', split, json_file)
    with open(label_path, 'w') as f:
        json.dump(bdd_label, f)

    return {
        'image': image_path,
        'label': label_path,
        'mask': mask_path,
        'lane': lane_path
    }


def convert_labelimg_to_bdd100k(input_dir, output_dir, val_ratio=0.2):
    create_bdd100k_dirs(output_dir)

    json_dir = os.path.join(input_dir, 'Annotations_json')
    img_dir = os.path.join(input_dir, 'JPEGImages')

    all_files = [f for f in os.listdir(json_dir) if f.endswith('.json')]
    np.random.shuffle(all_files)

    val_size = int(len(all_files) * val_ratio)
    train_files = all_files[val_size:]
    val_files = all_files[:val_size]

    gt_db = []

    print("Processing training set...")
    for json_file in tqdm(train_files):
        gt_db.append(process_file(json_file, json_dir, img_dir, output_dir, 'train'))

    print("Processing validation set...")
    for json_file in tqdm(val_files):
        gt_db.append(process_file(json_file, json_dir, img_dir, output_dir, 'val'))

    print('Database build finished')
    return gt_db


if __name__ == "__main__":
    input_dir = "dataset160"  # 包含Annotations_json和JPEGImages的目录
    output_dir = "bdd_demo"
    gt_db = convert_labelimg_to_bdd100k(input_dir, output_dir)

    # 打印一些统计信息
    total_images = len(gt_db)
    total_cars = sum(len(json.load(open(item['label']))['frames'][0]['objects']) for item in gt_db)
    print(f"Total processed images: {total_images}")
    print(f"Total detected cars: {total_cars}")
    print(f"Average cars per image: {total_cars / total_images:.2f}")