import numpy
import cv2
import os
import argparse
import sys
import math

parser = argparse.ArgumentParser(description='Dartboard detection')
parser.add_argument('-folder', '-f', type=str, default='Dartboard')
args = parser.parse_args()

cascade_name = "Dartboardcascade/cascade.xml"

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



def show_img(window, img):
    cv2.namedWindow(window, 0)
    cv2.resizeWindow(window, int(img.shape[1]), int(img.shape[0]))
    cv2.imshow(window, img)

def findCir_R(img, r, threshold):
    width = img.shape[1]
    height = img.shape[0]
    sumArr = [0] * (width * height)
    for y in range(height):
        for x in range(width):
            if img[y][x] != 255:
                continue
            for a in range(0, 360, 5):
                theta = a * math.pi / 180
                cx = int(x - r * math.cos(theta))
                cy = int(y - r * math.sin(theta))
                if cx > 0 and cx < width and cy > 0 and cy < height:
                    sumArr[cy * width + cx] += 1
        # Save Hough space image
    hough_space_img = numpy.array(sumArr).reshape(height, width)
    hough_space_img = cv2.normalize(hough_space_img, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(f'{args.folder}/hough_space_{i}.jpg', hough_space_img)

    circles = []
    for y in range(height):
        for x in range(width):
            if sumArr[y * width + x] >= threshold:
                circles.append((x, y, r))

    return circles

def findCir(img, rmin, rmax, threshold):
    circles = []
    for r in range(rmin, rmax + 1, 1):
        circles += findCir_R(img, r, threshold)
    return circles

def convolution(img, y, x, width, height, sobel):
    value = [0] * 9
    if y - 1 >= 0 and x - 1 >= 0 and y - 1 < height and x - 1 < width:
        value[0] = img[y - 1][x - 1]
    if y - 1 >= 0 and y - 1 < height:
        value[1] = img[y - 1][x]
    if y - 1 >= 0 and y - 1 < height and x + 1 < width:
        value[2] = img[y - 1][x + 1]
    if x - 1 >= 0:
        value[3] = img[y][x - 1]
    value[4] = img[y][x]
    if x + 1 < width:
        value[5] = img[y][x + 1]
    if y + 1 < height and x - 1 >= 0:
        value[6] = img[y + 1][x - 1]
    if y + 1 < height:
        value[7] = img[y + 1][x]
    if y + 1 < height and x + 1 < width:
        value[8] = img[y + 1][x + 1]
    out = abs(sum(value[i] * sobel[i] for i in range(9)))
    return int(out)

def houghdector(img, minVal, maxVal):
    sobel1 = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    sobel2 = [-1, 0, 1, -2, 0, 2, -1, 0, 1]
    width = img.shape[1]
    height = img.shape[0]
    out = numpy.zeros(img.shape, dtype=numpy.uint8)
    for y in range(height):
        for x in range(width):
            gx = convolution(img, y, x, width, height, sobel1)
            gy = convolution(img, y, x, width, height, sobel2)
            temp = gx + gy
            out[y][x] = min(max(temp, 0), 255)
    grad_mag_img = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite(f'{args.folder}/gradient_magnitude_{i}.jpg', grad_mag_img)

    return out

def detectAndDisplay(frame, image_name, groundtruth_boxes, model):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = findCir(houghdector(gray, 100, 200), 30, 100, 48)  # Detect circles using Hough Transform
    dartboards = model.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)  # Detect dartboards using Viola-Jones

    TP = 0  # True Positives
    FP = 0  # False Positives
    FN = len(groundtruth_boxes)  # False Negatives, initially set to the number of groundtruth boxes

    for (x, y, r) in circles:
        cv2.circle(frame, (x, y), r, (0, 0, 255), 2)  # Draw circles

    for (x, y, w, h) in dartboards:
        detected_box = (x, y, w, h)
        iou_max = 0
        for gt_box in groundtruth_boxes:
            iou = IOU(detected_box, gt_box)
            if iou > iou_max:
                iou_max = iou

        if iou_max > 0.5:  # Adjust the IOU threshold as needed
            TP += 1
            FN -= 1
        else:
            FP += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Draw dartboard bounding boxes

    # Draw red bounding boxes for ground truth
    for gt_box in groundtruth_boxes:
        x, y, w, h = gt_box
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return frame, TP, FP, FN


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

    cv2.imwrite(os.path.join(args.folder, f'dartboard_detected_hough_{i}.jpg'), frame_with_boxes)

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