from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import zipfile
import subprocess
from PIL import Image
import threading
from subprocess import Popen

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 确保设置一个秘密密钥
UPLOAD_FOLDER = 'uploads'
TEST_FOLDER = 'test'
RESULT_FOLDER = 'result'
UPLOAD_FOLDER = 'annotations'  # 目标文件夹
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 确保文件夹存在
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 处理状态
processing = False


def run_detection():
    global processing
    processing = True  # 标记为正在处理
    try:

        # 使用 subprocess.run() 运行 detect.py，并捕获输出
        result = subprocess.run(
            ["python", "detect.py"],
            check=True,  # 如果命令返回非零状态，则引发异常
            stdout=subprocess.PIPE,  # 捕获标准输出
            stderr=subprocess.PIPE,  # 捕获标准错误输出
            text=True  # 将输出作为字符串而非字节处理
        )

        # 打印 detect.py 的标准输出
        print("detect.py 输出:\n", result.stdout)

        # print("完成1")

    except subprocess.CalledProcessError as e:  # 捕获子进程返回的错误
        print(f"Subprocess error occurred: {e}")
        print(f"错误输出:\n{e.stderr}")  # 输出详细的错误信息
    except Exception as e:  # 捕获其他异常
        print(f"An unexpected error occurred: {e}")
    finally:
        # print("完成2")
        processing = False  # 标记为处理完成


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file and file.filename.endswith('.zip'):
        zip_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(zip_path)

        # 解压缩文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(TEST_FOLDER)

        # 转换所有图片为 PNG 格式
        for item in os.listdir(TEST_FOLDER):
            if item.lower().endswith(('.jpg', '.jpeg', '.bmp', '.gif', '.tiff')):
                img_path = os.path.join(TEST_FOLDER, item)
                img = Image.open(img_path)
                png_image_path = os.path.splitext(img_path)[0] + '.png'
                img.save(png_image_path, 'PNG')  # 保存为 PNG 格式
                os.remove(img_path)  # 删除原始文件

        # 执行 run_detection 函数
        # threading.Thread(target=run_detection).start()
        Popen(['python', 'detect.py'])
        # 返回处理中的页面
        return render_template('processing.html')

    return redirect(url_for('index'))


@app.route('/images')
def image_list():
    images = os.listdir(RESULT_FOLDER)  # 从结果文件夹读取处理后的图像
    return render_template('display.html', images=images)


@app.route('/check_processing')
def check_processing():
    # 检查是否完成run_detection，完成则将status置为completed
    print(processing)
    return jsonify({"status": "processing" if processing else "completed"})


@app.route('/result/<filename>')
def send_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)


@app.route('/annotations/<filename>')
def get_annotations(filename):
    txt_path = os.path.join(RESULT_FOLDER, filename.replace('.png', '.txt'))  # 假设图像为 .png 格式
    annotations = []
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                # 格式: <class_id> <x_center> <y_center> <width> <height>
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # 跳过格式不正确的行
                class_id, x_center, y_center, width, height = map(float, parts)

                # 将 YOLO 规范转换为标注记录
                annotations.append({
                    'category': int(class_id),  # 类别编号
                    'x': (x_center - width / 2) * 1024,  # 将相对坐标转换为绝对坐标
                    'y': (y_center - height / 2) * 1024,
                    'width': width * 1024,
                    'height': height * 1024
                })
    except FileNotFoundError:
        print(f"Annotations file not found: {txt_path}")
    return jsonify(annotations)


@app.route('/upload_annotations', methods=['POST'])
def upload_annotations():
    file_content = request.form['file_content']
    file_name = request.form['file_name']

    file_path = os.path.join(UPLOAD_FOLDER, f"{file_name}.txt")  # 组成完整的文件路径

    with open(file_path, 'w') as f:
        f.write(file_content)  # 保存内容

    return jsonify({"message": "文件已成功保存."})


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(TEST_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True)
