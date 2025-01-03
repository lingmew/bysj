import cv2
import os
from doclayout_yolo import YOLOv10

def process_images(model_path: str, input_folder: str, output_folder: str, imgsz: int = 1024, conf: float = 0.2, device: str = "cuda:0"):

    # 导入预训练模型
    model = YOLOv10(model_path)

    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹（如果不存在）

    # 遍历输入文件夹中的所有图片
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):  # 只处理特定格式的图片
            image_path = os.path.join(input_folder, filename)

            # Perform prediction
            det_res = model.predict(
                image_path,  # 图片路径
                imgsz=imgsz,  # 图片尺寸
                conf=conf,    # 置信率阈值
                device=device  # 设备使用 (e.g., 'cuda:0' or 'cpu')
            )

            # 标注结果
            annotated_frame = det_res[0].plot(pil=True, line_width=5, font_size=20)
            print(det_res[0].names)
            # 保存结果到指定的输出文件夹
            result_path = os.path.join(output_folder, filename)
            cv2.imwrite(result_path, annotated_frame)

    print("处理完成，结果已保存到:", output_folder)

# 如果需要直接运行此文件，可以使用以下代码
if __name__ == "__main__":
    model_path = "./model/doclayout_yolo_docstructbench_imgsz1024.pt"
    input_folder = "./datasets/example/"
    output_folder = "./results/"
    process_images(model_path, input_folder, output_folder)
