import os


def calculate_lap(box1, box2):
    # Convert YOLO to (x1, y1, x2, y2) format
    x1_min = box1[1] - box1[3] / 2
    y1_min = box1[2] - box1[4] / 2
    x1_max = box1[1] + box1[3] / 2
    y1_max = box1[2] + box1[4] / 2

    x2_min = box2[1] - box2[3] / 2
    y2_min = box2[2] - box2[4] / 2
    x2_max = box2[1] + box2[3] / 2
    y2_max = box2[2] + box2[4] / 2

    # Compute intersection
    inter_x1 = max(x1_min, x2_min)
    inter_y1 = max(y1_min, y2_min)
    inter_x2 = min(x1_max, x2_max)
    inter_y2 = min(y1_max, y2_max)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0  # No intersection

    intersection_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    lap = intersection_area / min(box1_area, box2_area)
    print(lap)
    return lap


def get_min_bbox(box1, box2):
    # 计算包裹两个框的最小外接框
    x1_min = min(box1[1] - box1[3] / 2, box2[1] - box2[3] / 2)
    y1_min = min(box1[2] - box1[4] / 2, box2[2] - box2[4] / 2)
    x2_max = max(box1[1] + box1[3] / 2, box2[1] + box2[3] / 2)
    y2_max = max(box1[2] + box1[4] / 2, box2[2] + box2[4] / 2)

    # 返回 (class_id, x_center, y_center, width, height)
    min_bbox = (box1[0], (x1_min + x2_max) / 2, (y1_min + y2_max) / 2, x2_max - x1_min, y2_max - y1_min)
    return min_bbox


def merge_overlapping_boxes(boxes, lap_threshold=0.8):
    # print("boxes:",boxes)
    n = len(boxes)
    indices = list(range(n))  # 初始化索引

    # 循环合并直到没有框可以合并
    while True:
        new_merge = False
        # 创建一个新的框列表用于合并
        new_boxes = boxes[:]
        n = len(new_boxes)
        for i in range(n):
            if indices[i] == -1:  # 如果框已经合并，跳过
                continue

            for j in range(i + 1, n):
                if indices[j] == -1:  # 如果框已经合并，跳过
                    continue

                # 判断两个框是否属于同一类
                if boxes[i][0] == boxes[j][0]:
                    lap = calculate_lap(boxes[i], boxes[j])

                    # 如果重叠度大于阈值，进行合并
                    if lap > lap_threshold:
                        new_merge = True
                        # 合并两个框
                        merged_box = get_min_bbox(boxes[i], boxes[j])

                        # 更新框的索引，标记已合并
                        indices.append(indices[i])
                        indices[i] = -1
                        indices[j] = -1

                        # 用合并后的框替换原来的两个框
                        new_boxes.append(merged_box)
                        # print("new box:",new_boxes)
                        break

            if new_merge:  # 如果合并了框，退出循环重新检查
                break

        # 如果本轮没有任何框被合并，说明合并完成
        if not new_merge:
            break

        # 更新框列表为合并后的新框
        boxes = new_boxes
        # print(boxes)
        print(indices)

    # 将未合并的框添加到最终框列表

    final_boxes = []
    index = []
    for i in range(len(indices)):
        if indices[i] != -1:
            final_boxes.append(boxes[i])
            index.append(indices[i])

    print(index)
    # 根据框的顺序进行排序
    combined = list(zip(final_boxes, index))
    combined.sort(key=lambda x: x[1])
    sort_boxes = [box for box, _ in combined]
    # print("final:",final_boxes)
    return sort_boxes


def process_boxes(file_path, lap_threshold=0.8):
    boxes = read_boxes_from_file(file_path)

    # 按照 class_id 分组检测框
    class_groups = {}
    for box in boxes:
        class_id = box[0]
        if class_id not in class_groups:
            class_groups[class_id] = []
        class_groups[class_id].append(box)

    # 对每个类的框进行合并
    final_boxes = []
    for class_id in sorted(class_groups.keys()):
        boxes_for_class = class_groups[class_id]
        merged_boxes = merge_overlapping_boxes(boxes_for_class, lap_threshold)
        final_boxes.extend(merged_boxes)  # 合并后的框加入最终结果

    return final_boxes


def read_boxes_from_file(file_path):
    boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            # 读取每一行，并解析检测框格式 (class_id, x_center, y_center, width, height)
            parts = list(map(float, line.strip().split()))
            # Ensure class_id is an integer
            parts[0] = int(parts[0])  # 强制转换 class_id 为整数
            boxes.append(tuple(parts))
    return boxes


def save_boxes_to_file(boxes, output_file_path):
    with open(output_file_path, 'w') as f:
        for box in boxes:
            f.write(' '.join(map(str, box)) + '\n')


def process_detection_files(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_folder, filename)
            boxes = read_boxes_from_file(input_file_path)
            merged_boxes = merge_overlapping_boxes(boxes, 0.6)
            output_file_path = os.path.join(output_folder, filename)
            save_boxes_to_file(merged_boxes, output_file_path)


if __name__ == "__main__":
    # 指定输入和输出文件夹路径
    input_folder = 'merge'
    output_folder = 'merged_results'

    process_detection_files(input_folder, output_folder)
