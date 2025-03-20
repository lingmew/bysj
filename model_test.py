import torch


def load_yolov10_model(model_path):
    """加载YOLOv10模型并返回模型及其结构信息"""
    # 加载模型
    model = torch.load(model_path, map_location=torch.device('cpu'))  # 如果有GPU,可以改为'cuda'

    # 如果模型是以state_dict的形式保存的
    if 'model' in model:
        # 可能需要根据YOLOv10具体实现寻找正确的key
        model = model['model']

    return model


def print_model_info(model):
    # 打印模型结构
    print("模型结构:")
    print(model)

    # 获取所有可学习的参数
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n可学习的参数总数: {num_params}")

    # 打印每个层的参数
    print("\n每层参数:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.size()}")


# 示例使用
if __name__ == '__main__':
    model_path = './model/doclayout_yolo_docstructbench_imgsz1024.pt'  # 替换为你的模型路径
    model = load_yolov10_model(model_path)
    print_model_info(model)
