
import numpy as np
import cv2
from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from collections import defaultdict
import matplotlib.pyplot as plt


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


def sobel(input_image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    # Sobel核在x方向上
    Kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    # Sobel核在y方向上
    Kernely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # 使用卷积函数应用Sobel核
    Ix = convolution(input_image, Kernelx)
    Iy = convolution(input_image, Kernely)

    # 计算梯度幅值
    gradient_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)
    # 计算梯度方向，添加一个小值epsilon防止除零错误
    epsilon = 1e-10
    gradient_direction = np.arctan2(Iy, Ix + epsilon)

    return Ix, Iy, gradient_magnitude, gradient_direction


# 从文件系统加载图像
image = cv2.imread('coins1.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否正确加载
if image is None:
    print("错误：未找到图像。")
else:
    # 应用Sobel算子
    Ix, Iy, gradient_magnitude, gradient_direction = sobel(image)

def hough_circle_detection(gradient_magnitude, rmin, rmax, steps, threshold):
    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = defaultdict(int)
    for x in range(gradient_magnitude.shape[0]):
        for y in range(gradient_magnitude.shape[1]):
            if gradient_magnitude[x, y] > 0:  # Check if it's an edge pixel
                for r, dx, dy in points:
                    a = x - dx
                    b = y - dy
                    acc[(a, b, r)] += 1

    circles = []
    for k, v in sorted(acc.items(), key=lambda i: -i[1]):
        x, y, r = k
        if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
            circles.append((x, y, r))

    return circles

# Load and process the image
image = cv2.imread('coins1.png', cv2.IMREAD_GRAYSCALE)
if image is None:
    print("Error: Image not found.")
else:
    Ix, Iy, gradient_magnitude, gradient_direction = sobel(image)

    # Perform Hough Circle Detection
    rmin = 30
    rmax = 75
    steps = 100
    threshold = 1.0
    circles = hough_circle_detection(gradient_magnitude, rmin, rmax, steps, threshold)

    # Draw the circles on the original image
    input_image = Image.fromarray(image).convert('RGB')
    draw_result = ImageDraw.Draw(input_image)
    for x, y, r in circles:
        draw_result.ellipse((y-r, x-r, y+r, x+r), outline=(255,0,0))

    # Save the output image
    input_image.save("result.png")

    # Show Sobel results
    plt.subplot(2, 2, 1), plt.imshow(Ix, cmap='gray'), plt.title('Ix')
    plt.subplot(2, 2, 2), plt.imshow(Iy, cmap='gray'), plt.title('Iy')
    plt.subplot(2, 2, 3), plt.imshow(gradient_magnitude, cmap='gray'), plt.title('Gradient Magnitude')
    plt.subplot(2, 2, 4), plt.imshow(gradient_direction, cmap='gray'), plt.title('Gradient Direction')
    plt.show()


