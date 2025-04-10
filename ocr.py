from paddleocr import PaddleOCR
import os
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
# 类别映射
category_mapping = {
    0: 'title', 1: 'plain text', 2: 'abandon', 3: 'figure', 4: 'figure_caption',
    5: 'table', 6: 'table_caption', 7: 'table_footnote', 8: 'isolate_formula', 9: 'formula_caption'
}

def save_text_results(images, category_data, directory='./results'):
    # 如果指定的目录不存在，则创建它
    if not os.path.exists(directory):
        os.makedirs(directory)

    for image, text in zip(images, category_data):
        # 形成文件名以保存的文本文件
        image_name = os.path.basename(image)
        text_filename = os.path.join(directory, f"{os.path.splitext(image_name)[0]}.txt")

        # 保存提取的文本到文件
        with open(text_filename, 'w', encoding='utf-8') as f:
            for title in text['title']:
                f.write(f"标题: {title}\n")
            for plain_text in text['plain_text']:
                f.write(f"正文: {plain_text}\n")
            for table in text['table']:
                f.write(f"表格内容: {table}\n")


def read_text_results(file_path):
    text_output = {
        'plain_text': [],
        'title': [],
        'table': [],
    }

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("标题:"):
                text_output['title'].append(line.replace("标题:", "").strip())
            elif line.startswith("正文:"):
                text_output['plain_text'].append(line.replace("正文:", "").strip())
            elif line.startswith("表格内容:"):
                text_output['table'].append(line.replace("表格内容:", "").strip())

    return text_output


def yolo2info(x_center, y_center, width, height, image):
    # 这里需要根据实际情况实现将 YOLO 格式坐标转换为图像坐标
    img_height, img_width = image.shape[:2]
    left = int((x_center - width / 2) * img_width)
    top = int((y_center - height / 2) * img_height)
    right = int((x_center + width / 2) * img_width)
    bottom = int((y_center + height / 2) * img_height)
    return left, top, right, bottom


def calculate_lap(box1, box2):
    """
    计算两个框的 LAP 值
    :param box1: 第一个框的信息 (category, x_center, y_center, width, height)
    :param box2: 第二个框的信息 (category, x_center, y_center, width, height)
    :return: LAP 值
    """
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2

    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2

    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    lap = intersection_area / min(box1_area, box2_area)
    return lap


def assign_ocr_to_annotations(ocr_boxes, annotations):
    """
    将 OCR 框分配到对应的标注框
    :param ocr_boxes: OCR 识别出的框列表，每个元素格式为 (0, x_center, y_center, width, height, text)
    :param annotations: 标注框列表，每个元素格式为 (category, x_center, y_center, width, height)
    :return: 分配结果字典，键为标注框索引，值为对应的 OCR 框文本内容列表
    """
    assigned = {i: [] for i in range(len(annotations))}
    lap_threshold = 0.5  # 设定重叠率阈值
    for ocr_box in ocr_boxes:
        for i, annotation in enumerate(annotations):
            lap = calculate_lap(ocr_box, annotation)
            # print(f"LAP between OCR box {ocr_box[5]} and annotation {i}: {lap}")
            if lap > lap_threshold:
                # 将 OCR 框的文本内容添加到对应的标注框结果中
                assigned[i].append(ocr_box)
    return assigned


def process_ocr_results(ocr_results, annotations, image):
    """
    处理 OCR 结果，将内容分配到 title 和 table 板块
    :param ocr_results: OCR 识别结果
    :param annotations: 标注框列表
    :param image: 图像
    :return: title 板块和 table 板块的处理结果
    """
    title_blocks = []
    table_blocks_list = []
    current_title = None
    current_table = []

    # 对标注框按位置排序
    sorted_annotations = sorted(annotations, key=lambda x: (x[2], x[1]))

    # 分配 OCR 结果到标注框
    assigned = assign_ocr_to_annotations(ocr_results, sorted_annotations)

    for index, ocr_boxes in assigned.items():
        if not ocr_boxes:  # 检查 ocr_boxes 是否为空
            continue
        annotation = sorted_annotations[index]
        category = annotation[0]
        # 待修改，根据标注框的类别来处理
        # print(f"Processing annotation {index} with category {category} has {ocr_boxes}")
        if category == 0:
            if current_title is not None:
                title_blocks.append(current_title)
            # 初始化 current_title
            current_title = {'title': [], 'content': []}
            for ocr_box in ocr_boxes:
                current_title['title'].append(ocr_box[5])
        elif category not in [5, 6, 7]:
            if current_title is None:
                current_title = {'title': ['无标题'], 'content': []}
            for ocr_box in ocr_boxes:
                current_title['content'].append(ocr_box[5])
        elif category == 5 or category == 6:  # table or table_caption
            for ocr_box in ocr_boxes:
                current_table.append(ocr_box)
        elif category == 7:  # table_footnote
            for ocr_box in ocr_boxes:
                current_table.append(ocr_box)
            table_blocks_list.append(current_table)
            current_table = []

    # 添加最后一个 title block
    if current_title is not None:
        title_blocks.append(current_title)
    # 添加最后一个 table block
    if current_table:
        table_blocks_list.append(current_table)

    return title_blocks, table_blocks_list


def generate_document(title_blocks, output_path):
    """
    生成文档文件
    :param title_blocks: title 板块的处理结果
    :param output_path: 文档文件输出路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for block in title_blocks:
            title = block['title'] if block['title'] else "默认标题"
            f.write(f"标题: {title}\n")
            f.write("内容:")
            for content in block['content']:
                f.write(f"{content}")
            f.write("\n")
        f.write("\n")


def generate_excel(table_blocks_list, base_name):
    """
    生成 Excel 文件，每个 table 单独保存一个 excel
    :param table_blocks_list: table 板块的处理结果列表
    :param base_name: 基础文件名
    """
    if not os.path.exists('ocr_result'):
        os.makedirs('ocr_result')
    for index, table_blocks in enumerate(table_blocks_list):
        if not table_blocks:
            continue
        # 初始化类点列表
        class_points = []
        # 第一个文本框的中心点作为第一个类点
        first_block = table_blocks[0]
        # print(f"first_block: {first_block}")
        _, x_center, _,  _, _, _ = first_block
        class_points.append(x_center)

        # 遍历其他文本框，检查其左右边界是否包含类点
        for block in table_blocks[1:]:
            _, x_center, _,  width, _, text = block
            left = x_center - width / 2
            right = x_center + width / 2
            contains_class_point = False
            for point in class_points:
                if left <= point <= right:
                    contains_class_point = True
                    break
            if not contains_class_point:
                class_points.append(x_center)

        # 对类点进行排序
        class_points.sort()
        
        # 为每个文本框分配列类
        column_classes = []
        for block in table_blocks:
            _, x_center, _, width, _, text = block
            for j, point in enumerate(class_points):
                
                if x_center-width/2 <= point <= x_center+width/2:
                    column_classes.append(j)
                    break
            else:
                column_classes.append(len(class_points) - 1)
        
        # 创建一个二维列表来存储表格数据
        max_columns = len(class_points)
        indice = 0
        transposed_data = [[] for _ in range(max_columns)]
        while indice < len(column_classes):
            for i in range(max_columns):
                if column_classes[indice] == i:
                    block = table_blocks[indice]
                    _, _, y_center, _, height, text = block
                    transposed_data[i].append(text)
                    indice += 1
                    if indice >= len(column_classes):
                        break
                else:
                    transposed_data[i].append("")
        df = pd.DataFrame(transposed_data).T
        excel_path = os.path.join('ocr_result', f'{base_name}_table_{index + 1}.xlsx')
        df.to_excel(excel_path, index=False, header=False)


def extract_text(image_path, annotations):
    """
    提取文本并生成文档和 Excel 文件
    :param image_path: 图像文件路径
    :param annotations: 标注框列表
    :return: 文档文件和 Excel 文件的路径，以及处理后的标题块和表格块信息
    """
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', det_model_dir='./model/ch_PP-OCRv4_det_infer')
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]  # 获取图像的高度和宽度
    results = ocr.ocr(image, cls=True)
    ocr_boxes = []
    for result in results:
        if result:
            for word_info in result:
                box = word_info[0]
                x_center = (box[0][0] + box[2][0]) / 2
                y_center = (box[0][1] + box[2][1]) / 2
                width = box[2][0] - box[0][0]
                height = box[2][1] - box[0][1]
                # 对坐标进行归一化处理
                x_center_normalized = x_center / img_width
                y_center_normalized = y_center / img_height
                width_normalized = width / img_width
                height_normalized = height / img_height
                text = word_info[1][0]
                ocr_boxes.append((0, x_center_normalized, y_center_normalized, width_normalized, height_normalized, text))

    title_blocks, table_blocks_list = process_ocr_results(ocr_boxes, annotations, image)

    # 生成文档
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    document_path = os.path.join('ocr_result', f'{base_name}.txt')
    if not os.path.exists('ocr_result'):
        os.makedirs('ocr_result')
    generate_document(title_blocks, document_path)

    # 生成 Excel 文件
    generate_excel(table_blocks_list, base_name)
    
    # 绘制 OCR 结果框到图像上
    for result in results:
        if result:
            for word_info in result:
                box = word_info[0]
                box = [(int(point[0]), int(point[1])) for point in box]
                cv2.polylines(image, [np.array(box)], isClosed=True, color=(0, 255, 0), thickness=2)
                text = word_info[1][0]
                cv2.putText(image, text, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # 保存绘制后的图像
    image_with_boxes_path = os.path.join('ocr_result', f'{base_name}_with_boxes.png')
    cv2.imwrite(image_with_boxes_path, image)
    
    # 处理标题块和表格块信息用于前端展示
    title_blocks_info = []
    for block in title_blocks:
        title = block['title'] if block['title'] else "默认标题"
        content = block['content']
        title_blocks_info.append({
            'title': title,
            'content': content
        })

    table_blocks_info = []
    for table_blocks in table_blocks_list:
        table_texts = []
        for table_content in table_blocks:
            table_texts.append(table_content[5])
        table_blocks_info.append(table_texts)

    return {
        'document_path': document_path,
        'excel_paths': [os.path.join('ocr_result', f'{base_name}_table_{i + 1}.xlsx') for i in range(len(table_blocks_list))],
        'title_blocks': title_blocks_info,
        'table_blocks': table_blocks_info
    }


def process_annotations(category_path, image):
    title_blocks = []
    table_blocks = []
    current_title = None

    with open(category_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            category, x_center, y_center, width, height = map(float, parts)
            category = int(category)
            left, top, right, bottom = yolo2info(x_center, y_center, width, height, image)

            if category == 0:  # title
                current_title = {
                    'title': '',
                    'content': [],
                    'top': top
                }
                title_blocks.append(current_title)
            elif current_title is not None and top > current_title['top']:
                current_title['content'].append((category, left, top, right, bottom))
            elif category in [5, 6, 7]:  # table, table_caption, table_footnote
                table_blocks.append((category, left, top, right, bottom))

    return title_blocks, table_blocks


if __name__ == '__main__':
    image_path = './test_data/0000.png'  # 替换为实际的图像文件路径
    annotations_path = './test_data/0000.txt'  # 替换为实际的标注文件路径
    annotations = []
    # 从 txt 文件中读取内容
    with open(annotations_path, 'r', encoding='utf-8') as file:
        annotations_str = file.read()
    for line in annotations_str.split('\n'):
        parts = line.strip().split()
        if len(parts) == 5:
            # 明确将 class_id 转换为 int 类型
            category = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
            annotations.append((category, x_center, y_center, width, height))
    result = extract_text(image_path, annotations)
    # print(result)
    
    