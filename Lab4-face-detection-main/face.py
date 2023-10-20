################################################
#
# COMS30068 - face.py
# University of Bristol
#
################################################

import numpy as np
import cv2
import os
import sys
import argparse

# LOADING THE IMAGE
# Example usage: python filter2d.py -n car1.png
parser = argparse.ArgumentParser(description='face detection')
parser.add_argument('-name', '-n', type=str, default='images/face1.jpg')
args = parser.parse_args()

# /** Global variables */
cascade_name = "frontalface.xml"


def IOU(box1, box2):
    x1, y1, width1, height1 = box1
    x2, y2, width2, height2 = box2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + width1, x2 + width2), min(y1 + height1, y2 + height2)
    intersection_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)  # 修改了变量名为intersection_area

    box1_area = width1 * height1
    box2_area = width2 * height2
    union_area = box1_area + box2_area - intersection_area

    # Check if union_area is zero
    if union_area == 0:
        return 0  # or return any other default value that makes sense in your context

    return intersection_area / union_area


def detectAndDisplay(frame, image_name):
    # image_basename_without_extension = os.path.splitext(os.path.basename(image_name))[0]
    frame, groundtruth_boxes = readGroundtruth(frame, os.path.splitext(os.path.basename(image_name))[0])


	# 1. Prepare Image by turning it into Grayscale and normalising lighting
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    # 2. Perform Viola-Jones Object Detection
    faces = model.detectMultiScale(frame_gray, scaleFactor=1.1, minNeighbors=1, flags=0, minSize=(10,10), maxSize=(300,300))
    # 3. Print number of Faces found
    print(len(faces))
    for face in faces:
        max_iou = 0
        for groundtruth_box in groundtruth_boxes:
            iou_value = IOU(face, groundtruth_box)
            max_iou = max(max_iou, iou_value)
        print(f"Max IOU for face {face} is {max_iou}")
    # 4. Draw box around faces found
    for i in range(0, len(faces)):
        start_point = (faces[i][0], faces[i][1])
        end_point = (faces[i][0] + faces[i][2], faces[i][1] + faces[i][3])
        colour = (0, 255, 0)
        thickness = 2
        frame = cv2.rectangle(frame, start_point, end_point, colour, thickness)

    TP = 0
    for groundtruth_box in groundtruth_boxes:
        detected = False
        for face in faces:
            iou_value = IOU(face, groundtruth_box)
            if iou_value > 0.5:  # 使用0.5的IOU阈值来考虑一个有效的检测
                detected = True
                break
        if detected:
            TP += 1

    FN = len(groundtruth_boxes) - TP
    FP = len (faces)-TP

    return TP, FN, FP  # 返回当前图像的TP和FN计数

# ************ NEED MODIFICATION ************
def readGroundtruth(frame, img_name, filename='groundtruth.txt'):
    groundtruth_boxes = []
    # read bounding boxes as ground truth
    with open(filename) as f:
        # read each line in text file
        for line in f.readlines():
            content_list = line.split(",")
            print(f"Checking for {img_name} in line: {line}")  # debug print
            if content_list[0] == img_name:
                x = int(float(content_list[1]))
                y = int(float(content_list[2]))
                width = int(float(content_list[3]))
                height = int(float(content_list[4]))
                groundtruth_boxes.append((x, y, width, height))
                # Draw the rectangle only if img_name matches
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 255), 2)
                print(f"Ground truth for {img_name}: x={x}, y={y}, width={width}, height={height}")
    return frame, groundtruth_boxes



# ==== MAIN ==============================================

imageName = args.name



# ignore if no such file is present.
if (not os.path.isfile(imageName)) or (not os.path.isfile(cascade_name)):
    print('No such file')
    sys.exit(1)

# 1. Read Input Image
frame = cv2.imread(imageName, 1)

# ignore if image is not array.
if not (type(frame) is np.ndarray):
    print('Not image data')
    sys.exit(1)


# 2. Load the Strong Classifier in a structure called `Cascade'
model = cv2.CascadeClassifier()
if not model.load(cascade_name): # if got error, you might need `if not model.load(cv2.samples.findFile(cascade_name)):' instead
    print('--(!)Error loading cascade model')
    exit(0)


TP_total = 0
FN_total = 0
FP_total = 0

# 3. 检测面部并显示结果
TP_current, FN_current, FP_current = detectAndDisplay(frame, imageName)
TP_total += TP_current
FN_total += FN_current
FP_total += FP_current


precision = TP_total / (TP_total + FP_total) if (TP_total +FP_total) !=0 else 0
recall = TP_total / (TP_total + FP_total) if(TP_total + FP_total)!=0 else 0
F1_score = 2 * (precision * recall) / (recall + precision)
print(f'F1-score is:{F1_score}')
# ... [代码的其余部分]

# 计算所有图像的TPR
TPR = TP_total / (TP_total + FN_total)
print(f"所有测试图像的真阳性率 (TPR) 为: {TPR}")



# 3. Detect Faces and Display Result
detectAndDisplay( frame, imageName)

# 4. Save Result Image
cv2.imwrite( "detected.jpg", frame )

