import cv2
import numpy as np

# 类别映射
class_mapping = {
    0: 'title',
    1: 'plain text',
    2: 'abandon',
    3: 'figure',
    4: 'figure_caption',
    5: 'table',
    6: 'table_caption',
    7: 'table_footnote',
    8: 'isolate_formula',
    9: 'formula_caption',
}

# 类别颜色 (使用BGR)
class_Colors = {
    0: (255, 0, 0),
    1: (200, 200, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
    4: (0, 255, 255),
    5: (0, 165, 255),
    6: (128, 0, 128),
    7: (147, 20, 255),
    8: (255, 255, 0),
    9: (0, 128, 128)
}


def draw_boxes(image_path, annotations_path, output_path):
    # 读取图片
    image = cv2.imread(image_path)

    # 打开YOLO格式的注释文件
    with open(annotations_path, 'r') as f:
        annotations = f.readlines()

    for idx, annotation in enumerate(annotations):
        # 解析YOLO格式：{class_id} {x_center} {y_center} {width} {height}
        parts = annotation.strip().split()
        class_id = int(parts[0])
        x_center, y_center, width, height = map(float, parts[1:])

        # 计算坐标
        img_height, img_width = image.shape[:2]
        x1 = int((x_center - width / 2) * img_width)
        y1 = int((y_center - height / 2) * img_height)
        x2 = int((x_center + width / 2) * img_width)
        y2 = int((y_center + height / 2) * img_height)

        # 获取颜色 (使用BGR而不是RGB)
        box_color = class_Colors[class_id]

        # 绘制框
        cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)

        # 添加类别标签和顺序ID
        label = f"{class_mapping[class_id]} {idx + 1}"

        # 设置字体和字体大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7  # 设置字体大小
        color = box_color
        thickness = 2  # 设置文本的粗细

        # 添加文本标签
        cv2.putText(image, label, (x1, y1 - 10), font, font_scale, color, thickness)

    # 保存生成的图像
    cv2.imwrite(output_path, image)


if __name__ == '__main__':
    input_file = 'test/0000.png'
    annotated_file = 'annotations/0000.txt'
    output_file = 'output/0000_result.png'
    draw_boxes(input_file, annotated_file, output_file)
