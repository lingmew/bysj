import cv2
import os
from doclayout_yolo import YOLOv10
from merge import merge_overlapping_boxes


def save_yolo_format(detections, output_path, img_width, img_height):
    oringin = []
    for det in detections:
        # 获取类别 ID 和边界框坐标
        class_id = det['class']  # 假设类别 ID 是整数
        bbox = det['bbox']  # 假设 bbox 格式为 [x_min, y_min, x_max, y_max]
        x_min, y_min, x_max, y_max = bbox

        # 计算边界框的中心点和宽高，并归一化
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        # 将数据以数字的形式加入列表
        oringin.append([class_id, x_center, y_center, width, height])

    boxes = oringin

    # 保存到文件
    with open(output_path, 'w') as f:
        for box in boxes:
            f.write(' '.join(map(str, box)) + '\n')


def process_images(model_path: str, input_folder: str, result_folder, output_folder: str, imgsz: int = 1024,
                   conf: float = 0.5,
                   device: str = "cuda:0"):
    # 导入预训练模型
    model = YOLOv10(model_path)

    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 只处理特定格式的图片
            image_path = os.path.join(input_folder, filename)

            # 读取图片以获取其大小
            img = cv2.imread(image_path)
            img_height, img_width = img.shape[:2]

            # 保存原始图像为 PNG 格式
            original_name = os.path.splitext(filename)[0] + '.png'
            original_path = os.path.join(result_folder, original_name)
            cv2.imwrite(original_path, img)

            # 进行预测
            det_res = model.predict(
                image_path,  # 图片路径
                imgsz=imgsz,  # 图片尺寸
                conf=conf,  # 置信率阈值
                device=device  # 设备使用 (e.g., 'cuda:0' or 'cpu')
            )

            # 处理预测结果
            annotations = []
            for detection in det_res[0].boxes.data:  # 访问结果的 raw data
                x1, y1, x2, y2 = map(int, detection[:4].tolist())  # 获取坐标并将其转换为列表
                class_id = int(detection[5])  # 第六个元素是类别 ID
                bbox = [x1, y1, x2, y2]  # 假设 bbox 是 [x_min, y_min, x_max, y_max]
                annotations.append({'class': class_id, 'bbox': bbox})

            # 保存 YOLO 格式标注文件
            yolo_label_path = os.path.join(result_folder, os.path.splitext(filename)[0] + '.txt')
            save_yolo_format(annotations, yolo_label_path, img_width, img_height)

            # 保存 YOLO 格式标注文件
            result_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.txt')
            save_yolo_format(annotations, result_path, img_width, img_height)
            '''
            # 标注结果
            annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)

            # 保存结果到指定的输出文件夹
            result_name = os.path.splitext(filename)[0] + '_result.png'
            result_path = os.path.join(output_folder, result_name)
            cv2.imwrite(result_path, annotated_frame)
            '''
    print("处理完成，标注结果已保存到:", output_folder)


# 如果需要直接运行此文件，可以使用以下代码
if __name__ == "__main__":
    model_path = "./model/doclayout_yolo_docstructbench_imgsz1024.pt"
    input_folder = './datasets/'
    output_folder = './datasets/'
    os.makedirs(output_folder, exist_ok=True)
    process_images(model_path, input_folder, output_folder, output_folder, 1024, 0.15)
