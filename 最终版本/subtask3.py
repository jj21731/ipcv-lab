import numpy as np
import cv2
import os
import argparse
import sys

# 设置命令行参数
parser = argparse.ArgumentParser(description='Dartboard detection')
parser.add_argument('-folder', '-f', type=str, default='Dartboard')
args = parser.parse_args()

# 设置分类器名称
cascade_name = "Dartboardcascade/cascade.xml"


# 形状分析函数
def shape_analysis(contours):
    dartboard_contours = []
    for cnt in contours:
        # 计算轮廓近似
        epsilon = 0.1 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 根据轮廓形状来识别飞镖靶（例如，圆形或接近圆形）
        if len(approx) > 8:
            dartboard_contours.append(cnt)
    return dartboard_contours


def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image


def scale_image(image, fx, fy):
    resized_image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_CUBIC)
    return resized_image


def convert_to_hsv(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return hsv_image


def apply_gaussian_blur(image, kernel_size=(5, 5)):
    blurred_image = cv2.GaussianBlur(image, kernel_size, 0)
    return blurred_image


def sharpen_image(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image


# IOU函数
def IOU(box1, box2):
    x1, y1, width1, height1 = box1
    x2, y2, width2, height2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + width1, x2 + width2), min(y1 + height1, y2 + height2)
    intersection_area = max(0, xi2 - yi1) * max(0, yi2 - yi1)

    box1_area = width1 * height1
    box2_area = width2 * height2
    union_area = box1_area + box2_area - intersection_area

    if union_area == 0:
        return 0

    return intersection_area / union_area


def hough_circle_detection(image, dp=1.2, minDist=100, param1=150, param2=50, minRadius=80, maxRadius=120,
                           maxCircles=50):
    # Convert image to grayscale if it is colored
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Gaussian Blurring of Gray Image
    blur_image = cv2.GaussianBlur(gray_image, (3, 3), 0)

    # Using OpenCV Canny Edge detector to detect edges
    edged_image = cv2.Canny(blur_image, param1, param2)

    height, width = edged_image.shape
    acc_array = np.zeros((height, width, maxRadius - minRadius))

    # Filling the accumulator array
    def fill_acc_array(x0, y0, radius):
        x = radius
        y = 0
        decision = 1 - x
        radius_index = radius - minRadius

        while y <= x:
            points = [
                (x + x0, y + y0), (y + x0, x + y0),  # Octants 1 and 2
                (-x + x0, y + y0), (-y + x0, x + y0),  # Octants 4 and 3
                (-x + x0, -y + y0), (-y + x0, -x + y0),  # Octants 5 and 6
                (x + x0, -y + y0), (y + x0, -x + y0)  # Octants 8 and 7
            ]
            for point in points:
                if 0 <= point[0] < height and 0 <= point[1] < width:
                    acc_array[point[0], point[1], radius_index] += 1

            y += 1
            if decision <= 0:
                decision += 2 * y + 1
            else:
                x -= 1
                decision += 2 * (y - x) + 1

    edges = np.where(edged_image == 255)
    for i in range(len(edges[0])):
        x = edges[0][i]
        y = edges[1][i]
        for radius in range(minRadius, maxRadius):
            fill_acc_array(x, y, radius)

    # Finding the circles
    detected_circles = []
    for i in range(height - 30):
        for j in range(width - 30):
            filter3D = acc_array[i:i + 30, j:j + 30, :] * 1
            max_pt = np.where(filter3D == filter3D.max())
            if filter3D.max() > 200:  # Threshold
                a = max_pt[0][0] + i
                b = max_pt[1][0] + j
                c = max_pt[2][0] + minRadius
                score = filter3D.max()
                detected_circles.append([b, a, c, score])

    # Sort and select the top circles
    detected_circles.sort(key=lambda x: x[3], reverse=True)  # Sort by score

    circles = np.array([detected_circles[:maxCircles]], dtype=np.float32) if len(detected_circles) > 0 else np.array([])

    return circles


# 检测并显示函数
def detectAndDisplay(frame, image_name, groundtruth_boxes, model):
    # 将图像转换为灰度并均衡化直方图
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)

    # 应用预处理
    frame_gray = apply_gaussian_blur(frame_gray)
    frame_gray = sharpen_image(frame_gray)

    # 使用霍夫变换检测圆形
    circles = hough_circle_detection(frame_gray)
    detected_circles = []
    print(len(circles.shape))
    if circles is not None and len(circles.shape) > 1:
        for circle in circles[0, :]:
            # center_x, center_y, radius, _ = circle
            center_x, center_y, radius = int(circle[0]), int(circle[1]), int(circle[2])
            x = int(center_x - radius)
            y = int(center_y - radius)
            w = h = int(2 * radius)
            detected_circles.append((x, y, w, h))  # 转换为边界框格式

            frame = cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 2)

    # 使用Viola-Jones检测飞镖靶
    dartboards = model.detectMultiScale(frame_gray)
    # 绘制由Viola-Jones检测到的飞镖靶边界框
    for (x, y, w, h) in dartboards:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 将两种检测结果合并
    all_detections = list(dartboards) + detected_circles

    for (x, y, w, h) in groundtruth_boxes:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # 使用红色画出边界框

    matched_groundtruth_boxes = set()
    TP = 0
    for detected_box in all_detections:
        max_iou = 0
        best_match = None
        for idx, groundtruth_box in enumerate(groundtruth_boxes):
            if idx not in matched_groundtruth_boxes:
                iou_value = IOU(detected_box, groundtruth_box)
                if iou_value > max_iou:
                    max_iou = iou_value
                    best_match = idx
        if max_iou > 0.5:  # IOU阈值
            TP += 1
            matched_groundtruth_boxes.add(best_match)

    FN = len(groundtruth_boxes) - len(matched_groundtruth_boxes)
    FP = len(all_detections) - TP

    # 形状分析
    contours, _ = cv2.findContours(frame_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    dartboard_contours = shape_analysis(contours)
    for cnt in dartboard_contours:
        x, y, w, h = cv2.boundingRect(cnt)
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

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
    print({
        "Image": image_name,
        "TP": TP_current,
        "FN": FN_current,
        "FP": FP_current,
        "Precision": precision_current,
        "TPR": TPR_current,
        "F1-Score": F1_score_current
    })
    cv2.imwrite(os.path.join(args.folder, f'dartboard_detected_improve_{i}.jpg'), frame_with_boxes)

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