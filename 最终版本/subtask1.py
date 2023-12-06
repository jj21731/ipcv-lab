import os
import argparse
import sys
import numpy
import cv2
import math

# 设置命令行参数
parser = argparse.ArgumentParser(description='Dartboard detection')
parser.add_argument('-folder', '-f', type=str, default='/Users/lil/Desktop/CW-I-Shape-Detection-Dartboard-main/Dartboard')
args = parser.parse_args()

# 设置分类器名称
cascade_name = "Dartboardcascade/cascade.xml"

# IOU函数
def IOU(box1, box2):
    x1, y1, width1, height1 = box1
    x2, y2, width2, height2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + width1, x2 + width2), min(y1 + height1, y2 + height2)
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = width1 * height1
    box2_area = width2 * height2
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area

# 检测和显示函数
def detectAndDisplay(frame, image_name, groundtruth_boxes, model):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    dartboards = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
    print(f'Number of Dartboards found in {image_name}: {len(dartboards)}')

    TP = 0
    for dartboard in dartboards:
        max_iou = 0
        for groundtruth_box in groundtruth_boxes:
            iou_value = IOU(dartboard, groundtruth_box)
            max_iou = max(max_iou, iou_value)
        if max_iou > 0.5:  # IOU阈值
            TP += 1
        # 绘制绿色边界框
        x, y, w, h = dartboard
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 绘制地面真相的红色边界框
    for gt_box in groundtruth_boxes:
        x, y, w, h = gt_box
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    FN = len(groundtruth_boxes) - TP
    FP = len(dartboards) - TP

    return frame, TP, FN, FP

# 读取地面真相数据
def readGroundtruth(img_name, filename='groundtruth.txt'):
    groundtruth_boxes = []
    with open(filename) as f:
        for line in f.readlines():
            content_list = line.split(",")
            if content_list[0] == img_name:
                x = int(float(content_list[1]))
                y = int(float(content_list[2]))
                width = int(float(content_list[3]))
                height = int(float(content_list[4]))
                groundtruth_boxes.append((x, y, width, height))
    return groundtruth_boxes

# 加载分类器模型
model = cv2.CascadeClassifier()
if not model.load(cascade_name):
    print('--(!)Error loading cascade model')
    sys.exit(1)

# 主程序处理图片
TP_total = 0
FN_total = 0
FP_total = 0
results = []

for i in range(16):  # dart0.jpg 到 dart15.jpg
    image_name = f'dart{i}.jpg'
    image_path = os.path.join(args.folder, image_name)

    if not os.path.isfile(image_path):
        print(f'File not found: {image_path}')
        continue

    frame = cv2.imread(image_path, 1)
    if not isinstance(frame, numpy.ndarray):
        print('Not image data:', image_path)
        continue

    groundtruth_boxes = readGroundtruth(f'dart{i}')
    frame_with_boxes, TP_current, FN_current, FP_current = detectAndDisplay(frame, image_name, groundtruth_boxes, model)
    TP_total += TP_current
    FN_total += FN_current
    FP_total += FP_current

    precision_current = TP_current / (TP_current + FP_current) if (TP_current + FP_current) != 0 else 0
    TPR_current = TP_current / (TP_current + FN_current) if (TP_current + FN_current) != 0 else 0
    F1_score_current = 2 * (precision_current * TPR_current) / (precision_current + TPR_current) if (
                                                                                                                      precision_current + TPR_current) != 0 else 0

    results.append({
        "Image": image_name,
        "TP": TP_current,
        "FN": FN_current,
        "FP": FP_current,
        "Precision": precision_current,
        "TPR": TPR_current,
        "F1-Score": F1_score_current
    })

    cv2.imwrite(os.path.join(args.folder, f'dartboard_detected_{i}.jpg'), frame_with_boxes)

# 计算总体性能指标
precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) != 0 else 0
TPR = TP_total / (TP_total + FN_total) if (TP_total + FN_total) != 0 else 0
F1_score = 2 * (precision * TPR) / (precision + TPR) if (precision + TPR) != 0 else 0
TPR = TP_total / (TP_total + FN_total) if (TP_total + FN_total) != 0 else 0

# 输出总体性能指标
print(f'\nTotal Performance:')
print(f"{'Total Precision':<20}: {precision:.2f}")
print(f"{'Total TPR':<20}: {TPR:.2f}")
print(f"{'Total F1-Score':<20}: {F1_score:.2f}")
print(f"{'Total TPR':<20}: {TPR:.2f}")

# 输出表格
print(f"\n{'Image':<12}{'TP':<6}{'FN':<6}{'FP':<6}{'Precision':<12}{'TPR':<12}{'F1-Score':<12}")
for res in results:
    print(
        f"{res['Image']:<12}{res['TP']:<6}{res['FN']:<6}{res['FP']:<6}{res['Precision']:<12.2f}{res['TPR']:<12.2f}{res['F1-Score']:<12.2f}")

# 输出平均性能指标
average_precision = sum(item["Precision"] for item in results) / len(results)
average_TPR = sum(item["TPR"] for item in results) / len(results)
average_F1_score = sum(item["F1-Score"] for item in results) / len(results)

print(f"\nAverage Performance for all 16 images:")
print(f"{'Average Precision':<20}: {average_precision:.2f}")
print(f"{'Average TPR':<20}: {average_TPR:.2f}")
print(f"{'Average F1-Score':<20}: {average_F1_score:.2f}")



