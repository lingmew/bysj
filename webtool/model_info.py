# 记录模型函数
import torch
import os

def load_model(file_path):
    model = torch.load(file_path, map_location=torch.device('cpu'))  # 加载模型
    return model


def get_model_info(MODEL_FOLDER):
    models_info = []
    for filename in os.listdir(MODEL_FOLDER):
        if filename.endswith('.pt'):  # 仅处理.pt文件
            model_path = os.path.join(MODEL_FOLDER, filename)  # 拼接模型路径
            # 获取模型的修改时间
            modified_time = os.path.getmtime(model_path)
            model = load_model(model_path)  # 加载模型
            # 性能查询
            accuracy = model['metrics']['accuracy'] if 'metrics' in model else 'N/A'
            loss = model['metrics']['loss'] if 'metrics' in model else 'N/A'
            # 添加模型信息到列表
            models_info.append({
                'name': filename,
                'accuracy': accuracy,
                'loss': loss,
                'modified_time': modified_time
            })

    return models_info