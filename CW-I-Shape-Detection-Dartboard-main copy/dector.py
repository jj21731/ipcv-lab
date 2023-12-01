
import os
import argparse
import sys
import numpy as np
import cv2
from math import sqrt, pi, cos, sin
from collections import defaultdict

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
    if not isinstance(frame, np.ndarray):
        print('Not image data:', image_path)
        continue

    groundtruth_boxes = readGroundtruth(f'dart{i}')
    frame_with_boxes, TP_current, FN_current, FP_current = detectAndDisplay(frame, image_name, groundtruth_boxes, model)
    TP_total += TP_current
    FN_total += FN_current
    FP_total += FP_current

    precision_current = TP_current / (TP_current + FP_current) if (TP_current + FP_current) != 0 else 0
    recall_current = TP_current / (TP_current + FN_current) if (TP_current + FN_current) != 0 else 0
    F1_score_current = 2 * (precision_current * recall_current) / (precision_current + recall_current) if (
                                                                                                                      precision_current + recall_current) != 0 else 0

    results.append({
        "Image": image_name,
        "TP": TP_current,
        "FN": FN_current,
        "FP": FP_current,
        "Precision": precision_current,
        "Recall": recall_current,
        "F1-Score": F1_score_current
    })

    cv2.imwrite(os.path.join(args.folder, f'dartboard_detected_{i}.jpg'), frame_with_boxes)

# 计算总体性能指标
precision = TP_total / (TP_total + FP_total) if (TP_total + FP_total) != 0 else 0
recall = TP_total / (TP_total + FN_total) if (TP_total + FN_total) != 0 else 0
F1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
TPR = TP_total / (TP_total + FN_total) if (TP_total + FN_total) != 0 else 0

# 输出总体性能指标
print(f'\nTotal Performance:')
print(f"{'Total Precision':<20}: {precision:.2f}")
print(f"{'Total Recall':<20}: {recall:.2f}")
print(f"{'Total F1-Score':<20}: {F1_score:.2f}")
print(f"{'Total TPR':<20}: {TPR:.2f}")

# 输出表格
print(f"\n{'Image':<12}{'TP':<6}{'FN':<6}{'FP':<6}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}")
for res in results:
    print(
        f"{res['Image']:<12}{res['TP']:<6}{res['FN']:<6}{res['FP']:<6}{res['Precision']:<12.2f}{res['Recall']:<12.2f}{res['F1-Score']:<12.2f}")

# 输出平均性能指标
average_precision = sum(item["Precision"] for item in results) / len(results)
average_recall = sum(item["Recall"] for item in results) / len(results)
average_F1_score = sum(item["F1-Score"] for item in results) / len(results)

print(f"\nAverage Performance for all 16 images:")
print(f"{'Average Precision':<20}: {average_precision:.2f}")
print(f"{'Average Recall':<20}: {average_recall:.2f}")
print(f"{'Average F1-Score':<20}: {average_F1_score:.2f}")


def convolution(input, kernel):
    # 初始化输出，使用输入图像的形状
    blurredOutput = np.zeros([input.shape[0], input.shape[1]], dtype=np.float32)

    # 计算核的半径
    kernelRadiusX = kernel.shape[0] // 2
    kernelRadiusY = kernel.shape[1] // 2

    # 创建输入图像的填充版本，避免边界效应
    paddedInput = cv2.copyMakeBorder(input, kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
                                     cv2.BORDER_REPLICATE)

    # 执行卷积
    for i in range(input.shape[0]):
        for j in range(input.shape[1]):
            # 提取与核对应的图像区域
            patch = paddedInput[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            # 计算区域和核的逐元素乘积之和
            sum = (np.multiply(patch, kernel)).sum()
            # 将卷积结果存入输出图像
            blurredOutput[i, j] = sum

    return blurredOutput


def sobel_optimized(input_image):
    # 转换为灰度图像
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # 高斯模糊以减少噪声
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Sobel核
    sobelx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=5)

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    # 计算梯度方向
    gradient_direction = np.arctan2(sobely, sobelx)

    return gradient_magnitude, gradient_direction


def hough_circle_detection(gradient_magnitude, rmin, rmax, steps, threshold):
    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = defaultdict(int)
    for x in range(gradient_magnitude.shape[0]):
        for y in range(gradient_magnitude.shape[1]):
            if gradient_magnitude[x, y] > threshold:
                for r, dx, dy in points:
                    a = x - dx
                    b = y - dy
                    acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold:
            circles.append((x, y, r))

    return circles
def combine_detections(frame, dartboards, circles):
    for dartboard in dartboards:
        (x, y, w, h) = dartboard
        for circle in circles:
            (cx, cy, cr) = circle
            if x < cx < x+w and y < cy < y+h:
                # 在这里，我们找到了一个圆形位于Viola-Jones探测到的区域内
                # 在这个区域内标记一个检测到的飞镖靶
                frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                break

    return frame


# 主程序处理图片
for i in range(1):  # 从 dartboard_detected_0.jpg 到 dartboard_detected_15.jpg
    image_name = f'dartboard_detected_{i}.jpg'
    image_path = os.path.join('/Users/lil/Desktop/CW-I-Shape-Detection-Dartboard-main/Dartboard', image_name)

    frame = cv2.imread(image_path)
    if frame is None:
        print(f'File not found: {image_path}')
        continue

    # 应用Viola-Jones探测器
    groundtruth_boxes = readGroundtruth(f'dart{i}')  # 确保这个函数读取正确的地面真相数据
    dartboards = detectAndDisplay(frame, image_name, groundtruth_boxes, model)

    # 应用霍夫圆变换
    gradient_magnitude, _ = sobel_optimized(frame)
    circles = hough_circle_detection(gradient_magnitude, 20, 100, 100, 50)  # 参数可能需要调整

    # 结合探测器结果
    result_frame = combine_detections(frame, dartboards, circles)

    # 显示或保存结果
    cv2.imshow(f'Dartboard Detection {i}', result_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 可选：保存结果图像
    cv2.imwrite(os.path.join('/Users/lil/Desktop/CW-I-Shape-Detection-Dartboard-main/Dartboard', f'result_{i}.jpg'), result_frame)
