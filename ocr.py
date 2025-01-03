import cv2
from paddleocr import PaddleOCR

def extract_text_from_image(image_path):
    # 初始化 OCR 模型
    ocr = PaddleOCR(use_angle_cls=True, lang='en') # 设置语言  
    
    result = ocr.ocr(image_path, cls=True)  # 调用 OCR 进行识别

    # 结果处理
    text_output = []
    for line in result:
        for word_info in line:
            text_output.append(word_info[1][0])  # 获取识别的文本
    
    return text_output

if __name__ == "__main__":
    image_path = './datasets/example/fuzzy_scan.jpg'  # 替换为你的图片路径
    extracted_text = extract_text_from_image(image_path)

    for text in extracted_text:
        print(text)

    # 可选择输出到文件
    with open('extracted_text.txt', 'w', encoding='utf-8') as f:
        for text in extracted_text:
            f.write(text + '\n')