import cv2
import os
import numpy as np

def preprocess_image(image):
    # 灰度化
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 提高对比度（可选，视情况而定）
    #img = cv2.equalizeHist(img)

    # 使用自适应阈值进行二值化
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)

    # 中值滤波去噪声
    img = cv2.medianBlur(img, 3)

    # 应用形态学闭运算
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)  # 闭运算
    #img = cv2.dilate(closing, kernel, iterations=1)  # 膨胀

    return img

def process_images_in_folder(folder_path, output_folder):
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的所有图像
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):  # 支持的文件扩展名
            img_path = os.path.join(folder_path, filename)
            # 读取图像
            image = cv2.imread(img_path)

            # 预处理图像
            processed_image = preprocess_image(image)

            # 保存预处理后的图像
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, processed_image)

if __name__ == "__main__":
    input_folder = "./datasets/test"  # 输入文件夹路径
    output_folder = "./datasets/test2"  # 输出文件夹路径

    process_images_in_folder(input_folder, output_folder)
    print(f"改善图像保存到 {output_folder}")
