import os
import detect

TEST_FOLDER = 'test'
RESULT_FOLDER = 'result'
MODEL_FOLDER = 'model'
model = 'doclayout_yolo_docstructbench_imgsz1024.pt'
model_path = os.path.join(MODEL_FOLDER, model)

detect.process_images(model_path, TEST_FOLDER, RESULT_FOLDER, 1024, 0.15)