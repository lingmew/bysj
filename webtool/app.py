from flask import Flask, render_template, request, redirect, jsonify, send_from_directory
import os
import zipfile
import subprocess
from PIL import Image
import datetime
from detect import process_images
from model_info import get_model_info
from waitress import serve
from ocr import extract_text  # 导入 ocr.py 中的函数
from merge import merge_overlapping_boxes
from plot import draw_boxes
from flask import Flask, request, jsonify


app = Flask(__name__)
app.secret_key = 'supersecretkey'  # 确保设置一个秘密密钥
UPLOAD_FOLDER = 'uploads'
TEST_FOLDER = 'test'
RESULT_FOLDER = 'result'
MODEL_FOLDER = 'model'
ANNOTATIONS_FOLDER = 'annotations'  # 目标文件夹
OCR_FOLDER = 'ocr_result'
OUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # 确保文件夹存在
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER


def clear_folder(folder):
    for item in os.listdir(folder):
        item_path = os.path.join(folder, item)
        if os.path.isdir(item_path):
            # 如果是文件夹，递归删除
            os.rmdir(item_path)  # 使用 rmtree(item_path) 如果需要删除非空文件夹
        else:
            # 如果是文件，直接删除
            os.remove(item_path)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    # 在解压前清空目标文件夹
    clear_folder(TEST_FOLDER)
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    model = request.form.get('selected_model')
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

        model_path = os.path.join(MODEL_FOLDER, model)
        process_images(model_path, TEST_FOLDER, RESULT_FOLDER, ANNOTATIONS_FOLDER, imgsz=1024, conf=0.1)

        # 清空解压包上传文件夹
        clear_folder(UPLOAD_FOLDER)
        return jsonify({"status": "success", "message": "图像处理完成!"})

    return jsonify({"status": "error", "message": "文件上传失败"}), 400


@app.route('/images')
def image_list():
    images = os.listdir(RESULT_FOLDER)  # 从结果文件夹读取处理后的图像
    return render_template('display.html', images=images)


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
                # 明确将 class_id 转换为 int 类型
                class_id = int(float(parts[0]))
                x_center, y_center, width, height = map(float, parts[1:])

                # 将 YOLO 规范转换为标注记录
                annotations.append({
                    'category': class_id,  # 类别编号
                    'x': (x_center - width / 2) * 1024,  # 将相对坐标转换为绝对坐标
                    'y': (y_center - height / 2) * 1024,
                    'width': width * 1024,
                    'height': height * 1024,
                    'visible': True
                })
    except FileNotFoundError:
        print(f"Annotations file not found: {txt_path}")
    return jsonify(annotations)


@app.route('/upload_annotations', methods=['POST'])
def upload_annotations():
    file_content = request.form['file_content']
    file_name = request.form['file_name']

    file_path = os.path.join(ANNOTATIONS_FOLDER, f"{file_name}.txt")  # 组成完整的文件路径

    with open(file_path, 'w') as f:
        f.write(file_content)  # 保存内容

    return jsonify({"message": "文件已成功保存."})


@app.route('/generate_result_image', methods=['POST'])
def generate_result_image():
    try:
        # 假设在annotations中有对应的.txt文件和test中有图片文件
        annotation_files = os.listdir(ANNOTATIONS_FOLDER)
        print("开始生成结果图")
        for annotation_file in annotation_files:
            # 获取图片文件路径
            image_name = annotation_file.replace('.txt', '.png')
            image_path = os.path.join(TEST_FOLDER, image_name)
            annotation_path = os.path.join(ANNOTATIONS_FOLDER, annotation_file)

            # 确定输出路径
            output_path = os.path.join(OUT_FOLDER, image_name)

            # 绘制结果图
            draw_boxes(image_path, annotation_path, output_path)
        print("结果图生成成功")
        return jsonify({"message": "结果图生成成功"})

    except Exception as e:
        return jsonify({"message": f"发生错误: {str(e)}"}), 500


@app.route('/extract_text/<image_name>', methods=['POST'])
def extract_text_route(image_name):
    image_path = os.path.join(TEST_FOLDER, image_name)
    annotations_str = request.form.get('annotations')
    annotations = []
    for line in annotations_str.split('\n'):
        parts = line.strip().split()
        if len(parts) == 5:
            # 明确将 class_id 转换为 int 类型
            category = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
            annotations.append((category, x_center, y_center, width, height))

    result = extract_text(image_path, annotations)
    return jsonify(result)


@app.route('/get_models', methods=['GET'])
def get_models():
    # 获取模型文件列表
    if not os.path.exists(MODEL_FOLDER):
        return jsonify([])  # 如果目录不存在，返回空列表

    models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.pt')]
    return jsonify(models)  # 返回JSON格式的模型列表


# 转化日期为可读形式
@app.template_filter('to_datetime')
def to_datetime(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


# 模型管理界面
@app.route('/model_management')
def model_management():
    models_info = get_model_info(MODEL_FOLDER)  # 获取模型信息
    return render_template('model_management.html', models=models_info)


@app.route('/upload_model', methods=['POST'])
def upload_model():
    if 'modelFile' not in request.files:
        return jsonify({"message": "没有选择文件"}), 400
    file = request.files['modelFile']
    if file.filename == '':
        return jsonify({"message": "没有选择文件"}), 400
    # 检查文件扩展名
    if not file.filename.endswith('.pt'):
        return jsonify({"message": "只支持上传 .pt 文件"}), 400
    # 保存文件
    file.save(os.path.join(app.config['MODEL_FOLDER'], file.filename))
    return jsonify({"message": "模型上传成功!"}), 200

@app.route('/model_training', methods=['POST'])
def model_training():
    data = request.get_json()

    # 生成训练命令
    command = "python train.py"
    for key, value in data.items():
        if isinstance(value, bool):
            if value:
                command += f" --{key}"
        elif value is not None:
            command += f" --{key} {value}"

    result = {
        "message": "训练请求已接收",
        "command": command
    }

    return jsonify(result)

@app.route('/batch_ocr', methods=['POST'])
def batch_ocr():
    annotations_folder = 'annotations'
    test_folder = 'test'
    for annotation_file in os.listdir(annotations_folder):
        if annotation_file.endswith('.txt'):
            image_name = annotation_file.replace('.txt', '.png')
            image_path = os.path.join(test_folder, image_name)
            annotation_path = os.path.join(annotations_folder, annotation_file)
            with open(annotation_path, 'r') as f:
                annotations = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        category, x_center, y_center, width, height = map(float, parts)
                        annotations.append((category, x_center, y_center, width, height))
            result = extract_text(image_path, annotations)
    return jsonify({"message": "批量 OCR 识别完成"})

@app.route('/training')
def training():
    model_name = request.args.get('model')
    return render_template('training.html', model=model_name)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(TEST_FOLDER, exist_ok=True)
    os.makedirs(RESULT_FOLDER, exist_ok=True)
    os.makedirs(ANNOTATIONS_FOLDER, exist_ok=True)
    os.makedirs(MODEL_FOLDER, exist_ok=True)
    # app.run(debug=True)
    print("服务器启用")
    serve(app, host='0.0.0.0', port=5001)
    print("服务器关闭")
