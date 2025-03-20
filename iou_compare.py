import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置 matplotlib 图表的默认字体为 SimHei（支持中文）
plt.rcParams['font.family'] = 'SimHei'


def calculate_iou(box1, box2):
    # 将 YOLO 格式的框转换为 (x1, y1, x2, y2) 格式
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2

    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2

    # 计算交集的坐标
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    # 如果没有交集，返回 IoU 为 0
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    # 计算交集面积
    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # 计算 IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    # iou = intersection_area / min(box1_area, box2_area)
    return iou


def read_yolo_format(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            # 检查框的宽度和高度是否为0，若是，打印警告
            if width == 0 and height == 0:
                print(f"Warning: Found a box with width and height 0 in file: {file_path}")

            boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def calculate_TFP(gt_boxes, det_boxes, iou_threshold=0.5):
    gt_finished = []
    det_flag = []
    # 遍历检测框
    for det_box in det_boxes:
        best_iou = 0
        best_gt_index = -1

        # 比较每个检测框与所有真实框的 IoU，找到最优的匹配
        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_box[0] == det_box[0] and gt_index not in gt_finished:
                iou = calculate_iou(gt_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_index = gt_index

        # 如果 IoU 满足阈值，则认为是一个匹配
        if best_iou >= iou_threshold:
            det_flag.append(1)  # TP
            gt_finished.append(best_gt_index)
        else:
            det_flag.append(0)  # FP

    return det_flag


def calculate_PR(gt_num, det_flag):
    TP = 0
    FP = 0
    precision = []
    recall = []
    for f in det_flag:
        if f == 0:
            FP += 1
        elif f == 1:
            TP += 1
        precision.append(TP / (TP + FP))
        recall.append(TP / gt_num)

    return precision, recall


def calculate_AP(num):
    L = len(num)
    for i in range(L - 1, 0, -1):
        if num[i - 1] < num[i]:
            num[i - 1] = num[i]
    num = np.array(num)
    avg = np.nanmean(num) if num.size > 0 and np.any(~np.isnan(num)) else 0.0
    return num, avg


def plot_pr(precisions, recalls, save_path, avg_ap):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    y = len(precisions)
    for i in range(y):
        ax.plot(recalls[i], i, precisions[i], label=f'Line {i + 1}')
        # 添加 APs 平均值到图上
    ax.text(-0.5, -0.5, 0.9, f"APs 平均值: {avg_ap:.4f}", horizontalalignment='center', verticalalignment='center',
            fontsize=12, color='red')
    ax.set_title("精度-召回率曲线")
    ax.set_xlabel("召回率 (Recall)")
    ax.set_zlabel("精度 (Precision)")
    ax.set_ylabel("y")
    ax.set_xlim([0.0, 1.0])
    ax.set_zlim([0.0, 1.0])

    # 保存图像
    plt.savefig(save_path)
    plt.close()  # 关闭图表


if __name__ == '__main__':
    compare_folder = './compare'
    gt_folder = os.path.join(compare_folder, '0')
    gt_boxes = {filename: read_yolo_format(os.path.join(gt_folder, filename))
                for filename in os.listdir(gt_folder) if filename.endswith('.txt')}
    for i in range(1, 10):
        det_folder = os.path.join(compare_folder, str(i))
        if not os.path.exists(det_folder):
            continue
            # 遍历检测文件
        print("fold:", i)
        precisions = []
        recalls = []
        APs = []
        for filename in os.listdir(det_folder):
            if filename.endswith('.txt') and filename in gt_boxes:
                det_boxes = read_yolo_format(os.path.join(det_folder, filename))
                det_flag = calculate_TFP(gt_boxes[filename], det_boxes, 0.5)
                precision, recall = calculate_PR(len(gt_boxes[filename]), det_flag)
                precision, AP = calculate_AP(precision)
                precisions.append(precision)
                recalls.append(recall)
                APs.append(AP)

        print(np.nanmean(APs))
        avg_ap = np.nanmean(APs)
        # 创建图像保存路径
        save_path = os.path.join(compare_folder, f"pr_{i}.png")
        plot_pr(precisions, recalls, save_path, avg_ap)
