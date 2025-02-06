import cv2
import os
from doclayout_yolo import YOLOv10

# 类别字典定义
category_dict = {
    0: 'title',
    1: 'plain_text', 
    2: 'abandon', 
    3: 'figure', 
    4: 'figure_caption', 
    5: 'table', 
    6: 'table_caption', 
    7: 'table_footnote', 
    8: 'isolate_formula', 
    9: 'formula_caption'
}

def load_model(model_path: str):
    """加载YOLOv10模型。"""
    return YOLOv10(model_path)

def save_cropped_image(cropped_image, output_folder, image_name, category_name, index):
    """保存裁剪后的图像到指定类别文件夹，支持同一类别多个裁剪图。"""
    # 创建以图片名称命名的文件夹
    image_folder = os.path.join(output_folder, image_name)
    os.makedirs(image_folder, exist_ok=True)  # 创建图片文件夹（如果不存在）

    # 创建类别文件夹（使用类别名称）
    category_folder = os.path.join(image_folder, category_name)  # 使用类别名称而不是 ID
    os.makedirs(category_folder, exist_ok=True)  # 创建类别文件夹（如果不存在）

    # 构建结果路径，使用索引确保名字唯一
    result_path = os.path.join(category_folder, f"{image_name}_crop_{category_name}_{index}.jpg")
    cv2.imwrite(result_path, cropped_image)

def process_image(model, image_path: str, output_folder: str):
    """处理单张图片进行目标检测和裁剪。"""
    image = cv2.imread(image_path)

    # Perform prediction
    det_res = model.predict(
        image_path,  # 图片路径
        imgsz=1024,  # 图片尺寸
        conf=0.2,    # 置信率阈值
        device="cuda:0"  # 设备使用 (e.g., 'cuda:0' or 'cpu')
    )

    # 获取图片名称（不包含扩展名）
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    # 按类别保存截取的图片
    index_dict = {}  # 用于跟踪每个类别的索引
    for detection in det_res[0].boxes.data:  # 访问结果的 raw data
        x1, y1, x2, y2 = map(int, detection[:4].tolist())  # 获取坐标并将其转换为列表
        category_id = int(detection[5])  # 第六个元素是类别 ID
        
        # 获取类别名称
        category_name = category_dict.get(category_id, f"unknown_{category_id}")  # 如果没有找到，则使用 "unknown"

        # 裁剪图像
        cropped_image = image[y1:y2, x1:x2]

        # 更新索引
        if category_name not in index_dict:
            index_dict[category_name] = 0
        
        # 保存裁剪后的图像，传递类别名称
        save_cropped_image(cropped_image, output_folder, image_name, category_name, index_dict[category_name])
        
        # 增加索引
        index_dict[category_name] += 1

def process_images(model_path: str, input_folder: str, output_folder: str):
    """处理输入文件夹中的所有图片。"""
    model = load_model(model_path)

    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹（如果不存在）

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 只处理特定格式的图片
            image_path = os.path.join(input_folder, filename)
            process_image(model, image_path, output_folder)

    print("处理完成，结果已保存到:", output_folder)

# 如果需要直接运行此文件，可以使用以下代码
if __name__ == "__main__":
    model_path = "./model/doclayout_yolo_docstructbench_imgsz1024.pt"
    input_folder = "./datasets/test/"
    output_folder = "./test_results_split/"
    process_images(model_path, input_folder, output_folder)
