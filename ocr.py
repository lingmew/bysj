import cv2
from paddleocr import PaddleOCR
import os


def extract_text_from_image(image_path):
    """从给定图像中提取文本"""
    ocr = PaddleOCR(use_angle_cls=True, lang='ch', det_model_dir='./model/ch_PP-OCRv4_det_infer')  # 初始化 OCR 模型
    result = ocr.ocr(image_path, cls=True)  # 调用 OCR 进行识别
    print(result)
    # 结果处理
    text_output = []
    if result:  # 检查结果是否有效
        for line in result:
            for word_info in line:
                text_output.append(word_info[1][0])  # 获取识别的文本

    return text_output


def save_text_results(text_results, output_folder, image_name, category_name):
    """将每个类别的识别结果保存到文本文件中"""
    text_file_path = os.path.join(output_folder, f"{image_name}_extracted_text.txt")

    with open(text_file_path, 'a', encoding='utf-8') as f:  # 以附加模式打开文件
        for text in text_results:
            f.write(text + '\n')
        f.write('\n')  # 添加空行分隔不同类别


def process_results_folder(results_folder):
    """遍历结果文件夹中的每个子文件夹，并对指定类别图片执行 OCR"""
    # 定义需要处理的类别
    categories_to_process = ['plain_text', 'figure_caption', 'table_caption', 'formula_caption']

    # 遍历所有图片文件夹
    for image_folder in os.listdir(results_folder):
        folder_path = os.path.join(results_folder, image_folder)

        if os.path.isdir(folder_path):  # 确保是一个文件夹
            # 遍历文件夹中的每个类别
            for category in categories_to_process:
                category_path = os.path.join(folder_path, category)

                if os.path.isdir(category_path):  # 确保类别文件夹存在
                    # 遍历类别文件夹中的每个图片文件
                    for filename in os.listdir(category_path):
                        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 只处理特定格式的图片
                            image_path = os.path.join(category_path, filename)
                            extracted_text = extract_text_from_image(image_path)

                            # 保存文本结果到对应图片的文件夹中
                            save_text_results(extracted_text, category_path, filename.split('.')[0], category)


if __name__ == "__main__":
    results_folder = './results/'    # 结果文件夹的路径
    process_results_folder(results_folder)
