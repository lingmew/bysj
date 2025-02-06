import streamlit as st
import zipfile
import os
import cv2
from paddleocr import PaddleOCR
import ocr
import split

# 定义自定义 CSS
custom_css = """
<style>
    body {
        background-color: #f0f0f5;
        font-family: 'Arial', sans-serif;
        color: #333;
    }

    h1 {
        text-align: center;
        color: #0056b3;
        font-size: 2.5em;
        margin-bottom: 20px;
    }

    .stButton {
        background-color: #008cba; 
        color: white; 
        padding: 10px 20px; 
        border: none; 
        border-radius: 5px; 
        cursor: pointer;
    }

    .stButton:hover {
        background-color: #005f7d;
    }

    .stFileUploader {
        margin: 20px auto;
        text-align: center;
        border: 2px dashed #008cba;
        padding: 40px;
        border-radius: 10px;
        background-color: #fff;
    }

    .success {
        color: green;
        font-weight: bold;
    }

    .error {
        color: red;
        font-weight: bold;
    }
</style>
"""

# 注入自定义 CSS 至 Streamlit 应用
st.markdown(custom_css, unsafe_allow_html=True)


def extract_zip(uploaded_file, output_folder):
    """解压缩上传的ZIP文件到指定的输出文件夹"""
    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(output_folder)


def image2txt():
    st.title("OCR Document Layout Analysis")
    st.write("请上传包含图片的压缩包（zip）:")

    uploaded_file = st.file_uploader("选择一个 ZIP 文件", type=["zip"])

    if uploaded_file is not None:
        # 创建临时文件夹
        temp_dir_images = "./temp_images"
        temp_dir_results = "./temp_results"
        os.makedirs(temp_dir_images, exist_ok=True)
        os.makedirs(temp_dir_results, exist_ok=True)

        # 解压缩文件
        extract_zip(uploaded_file, temp_dir_images)
        st.success("文件解压成功！")

        # 显示解压后的图片列表
        image_files = [f for f in os.listdir(temp_dir_images) if f.endswith(('.png', '.jpg', '.jpeg'))]
        selected_image = st.selectbox("选择一张图片查看识别结果", image_files)

        if st.button("开始解压并处理图片"):
            # 进行版面分析处理
            doclayout_model_path = "./model/doclayout_yolo_docstructbench_imgsz1024.pt"
            split.process_images(doclayout_model_path, temp_dir_images, temp_dir_results)

            # 进行ocr识别
            ocr.process_results_folder(temp_dir_results)
            st.success("分析成功！")

        if selected_image:
            # 显示原图
            image_path = os.path.join(temp_dir_images, selected_image)
            st.image(image_path, caption='原图', use_column_width=True)

            # 显示 OCR 识别结果
            result_path = os.path.join(temp_dir_results, f"{selected_image}.txt")  # 假设识别结果是以原图文件名+`.txt`命名
            if os.path.exists(result_path):
                with open(result_path, 'r', encoding='utf-8') as result_file:
                    ocr_result = result_file.read()
                st.text_area("OCR 识别结果", value=ocr_result, height=300)
            else:
                st.warning("未找到该图像的OCR识别结果。")


if __name__ == "__main__":
    image2txt()
