import os
import cv2
import json
import numpy as np


def mask_to_coco_with_vis(
        mask_dir,
        output_json,
        vis_dir=None,
        category_id=1,
        epsilon_factor=0.002,
        min_area=10,
        num_vis=20  # 最多可视化多少张图片
):
    """
    将二值化分割掩码图片转换为 COCO 格式的 JSON，并提供顶点稀疏度的可视化

    参数:
        mask_dir: 掩码图片所在文件夹路径 (掩码应为黑白图, 目标区域 > 0)
        output_json: 输出的 json 文件路径
        vis_dir: 可视化输出目录，若为 None 则不进行可视化
        category_id: 类别 ID
        epsilon_factor: 多边形拟合的精度系数，控制点稀疏程度
        min_area: 过滤面积噪点
        num_vis: 最多生成的可视化图像数量，防止数据量太大耗时过长
    """

    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": category_id, "name": "target_class"}]
    }

    annotation_id = 0
    image_id = 0
    vis_count = 0

    # 确保可视化目录存在
    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    filenames = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(valid_extensions)])

    for filename in filenames:
        mask_path = os.path.join(mask_dir, filename)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            continue

        height, width = mask.shape

        # 1. 记录 Image 信息
        coco_format["images"].append({
            "id": image_id,
            "file_name": filename,
            "width": width,
            "height": height
        })

        # 二值化
        _, binary_mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)

        # 用于可视化的彩色背景图（将灰度 mask 转为 BGR 方便画彩色点线）
        vis_image = None
        if vis_dir is not None and vis_count < num_vis:
            vis_image = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        # 2. 提取轮廓
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            # if area < min_area:
            #     continue

            # 3. 多边形拟合（提取稀疏顶点）
            epsilon = epsilon_factor * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            flattened_points = approx.flatten().tolist()
            if len(flattened_points) < 6:
                continue

            # 4. 获取 BBox
            x, y, w, h = cv2.boundingRect(approx)

            # 5. 构建 Annotation
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": [flattened_points],
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(area),
                "iscrowd": 1,
                "file_name": filename
            }
            coco_format["annotations"].append(annotation)
            annotation_id += 1

            # 6. 绘制可视化图
            if vis_image is not None:
                # 画出多边形的连线（蓝色），thickness=1
                cv2.drawContours(vis_image, [approx], -1, (255, 0, 0), 1)

                # 画出每一个被提取的顶点（红色实心圆），radius=3
                for point in approx:
                    px, py = point[0]
                    cv2.circle(vis_image, (px, py), 3, (0, 0, 255), -1)

        # 保存可视化图像
        if vis_image is not None:
            vis_save_path = os.path.join(vis_dir, f"vis_{filename}")
            cv2.imwrite(vis_save_path, vis_image)
            vis_count += 1

        image_id += 1

    # 写入 JSON
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(coco_format, f, ensure_ascii=False)

    print(f"转换完成！处理了 {image_id} 张图片，共提取了 {annotation_id} 个多边形标注。")
    print(f"JSON文件已保存在: {output_json}")
    if vis_dir is not None:
        print(f"可视化效果图已保存在: {vis_dir} (共 {vis_count} 张)")


if __name__ == "__main__":
    # 配置你的路径（建议统一为 256x256 切片后的 mask 路径）
    MASK_DIRECTORY = "/workspace/data/ht_dataset/r19c2/crops_test_256/labels"
    OUTPUT_JSON_PATH = "/workspace/data/ht_dataset/r19c2/crops_test_256/annotations/test.json"
    VIS_DIRECTORY = "/workspace/data/ht_dataset/r19c2/crops_test_256/vis_check"

    os.makedirs(os.path.dirname(OUTPUT_JSON_PATH), exist_ok=True)

    # 执行转换并生成可视化图像
    mask_to_coco_with_vis(
        mask_dir=MASK_DIRECTORY,
        output_json=OUTPUT_JSON_PATH,
        vis_dir=VIS_DIRECTORY,
        category_id=1,
        epsilon_factor=0.005,  # 核心参数：根据可视化图片中的红点密度进行调整
        min_area=10,
        num_vis=30  # 生成前 30 张切片的可视化用于检查
    )