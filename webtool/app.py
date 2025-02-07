from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import zipfile
import subprocess
import threading
from subprocess import Popen

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 确保设置一个秘密密钥
UPLOAD_FOLDER = 'uploads'
TEST_FOLDER = 'test'
RESULT_FOLDER = 'result'

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

        print("完成1")

    except subprocess.CalledProcessError as e:  # 捕获子进程返回的错误
        print(f"Subprocess error occurred: {e}")
        print(f"错误输出:\n{e.stderr}")  # 输出详细的错误信息
    except Exception as e:  # 捕获其他异常
        print(f"An unexpected error occurred: {e}")
    finally:
        print("完成2")
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


@app.route('/generate-json', methods=['POST'])
def generate_json():
    annotations = request.json
    with open('annotations.json', 'w') as json_file:
        json.dump(annotations, json_file, indent=4)
    return jsonify({'status': 'success'})


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(TEST_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    app.run(debug=True)
