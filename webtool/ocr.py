from paddleocr import PaddleOCR
import os
import cv2
from merge import calculate_iou


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
    h, w = image.shape[:2]
    x_center *= w
    y_center *= h
    width *= w
    height *= h
    left = int(x_center - width / 2) - 1
    top = int(y_center - height / 2) - 1
    right = int(x_center + width / 2) + 1
    bottom = int(y_center + height / 2) + 1
    # 检查裁剪区域是否在图像尺寸之内
    if left < 0: left = 0
    if top < 0: top = 0
    if right > image.shape[1]: right = image.shape[1]  # 横坐标
    if bottom > image.shape[0]: bottom = image.shape[0]  # 纵坐标
    return left, top, right, bottom


# def process_ocr(image_path, category_path):
#     ocr = PaddleOCR(use_angle_cls=True, lang='ch', det_model_dir='./model/ch_PP-OCRv4_det_infer')  # 初始化 OCR 模型
#     text_output = {
#         'plain_text': [],
#         'title': [],
#         'table': [],
#     }
#     image = cv2.imread(image_path)
#     # height, width = image.shape[:2]
#     with open(category_path, 'r') as f:
#         for line in f:
#             parts = line.strip().split()
#             if len(parts) != 5:
#                 continue  # 跳过格式不正确的行
#             category, x_center, y_center, width, height = map(float, parts)
#             category = int(category)
#             left, top, right, bottom = yolo2info(x_center, y_center, width, height, image)
#             img = image[top:bottom, left:right]
#             results = ocr.ocr(img, cls=True)  # 调用 OCR 进行识别
#             if results:
#                 for result in results:
#                     if result:
#                         for word_info in result:
#                             word_text = word_info[1][0]  # 获取识别出的文本
#                             if category == 0:  # 'title'
#                                 text_output['title'].append(word_text)
#                             elif category == 1:  # 'plain text'
#                                 text_output['plain_text'].append(word_text)
#                             elif category in [5, 6, 7]:  # 'table', 'table_caption', 'table_footnote'
#                                 text_output['table'].append(word_text)
#     return text_output


def process_ocr(image_path, category_path):
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', det_model_dir='./model/ch_PP-OCRv4_det_infer')  # 初始化 OCR 模型

    image = cv2.imread(image_path)
    results = ocr.ocr(image, cls=True)  # 调用 OCR 进行识别
