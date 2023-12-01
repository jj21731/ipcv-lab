import cv2
import matplotlib.pyplot as plt


def convolution2d(image, kernel):
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # 填充图像
    image_padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # 执行卷积操作并裁剪回原始大小
    output = np.zeros_like(image, dtype=np.float32)  # 保证输出图像大小与原图相同
    for x in range(pad_height, image.shape[0] + pad_height):
        for y in range(pad_width, image.shape[1] + pad_width):
            output[x - pad_height, y - pad_width] = (kernel * image_padded[x - pad_height:x + pad_height + 1, y - pad_width:y + pad_width + 1]).sum()

    return output



def sobel_filters(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = convolution2d(image, Kx)
    Iy = convolution2d(image, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return (G, theta)


def threshold_image(image, threshold):
    return (image > threshold) * 255


import numpy as np


def hough_circle_transform_optimized(image, magnitude, direction, threshold, min_radius, max_radius):
    rows, cols = image.shape
    radius_range = max_radius - min_radius
    hough_space = np.zeros((rows, cols, radius_range))

    # 获取所有大于阈值的坐标
    edges = np.argwhere(magnitude >= threshold)

    # 对于每个边缘点，计算所有可能的圆心
    for x, y in edges:
        for radius in range(radius_range):
            r = radius + min_radius
            # 计算圆周上的点坐标
            circle_perimeter = [(int(x - r * np.cos(t * np.pi / 180)), int(y - r * np.sin(t * np.pi / 180))) for t in
                                range(0, 360, 5)]
            # 更新霍夫空间
            for a, b in circle_perimeter:
                if 0 <= a < rows and 0 <= b < cols:
                    hough_space[a, b, radius] += 1
    return hough_space


def detect_circles(hough_space, threshold):
    detected_circles = []
    for radius in range(hough_space.shape[2]):
        layer = hough_space[:, :, radius]
        rows, cols = np.where(layer > threshold)
        for r, c in zip(rows, cols):
            detected_circles.append((r, c, radius + min_radius))
    return detected_circles


# 读取图像并转换为灰度
image = cv2.imread('coins1.png', cv2.IMREAD_GRAYSCALE)

# Sobel 边缘检测
gradient_magnitude, gradient_direction = sobel_filters(image)

# 应用阈值
threshed_magnitude = threshold_image(gradient_magnitude, 50)

# 霍夫圆变换
min_radius = 29
max_radius = 100
hough_space = hough_circle_transform_optimized(image, threshed_magnitude, gradient_direction, 50, min_radius, max_radius)

# 检测圆
detected_circles = detect_circles(hough_space, 100)

# 在原始图像上画出检测到的圆
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
for circle in detected_circles:
    cv2.circle(output_image, (circle[1], circle[0]), circle[2], (0, 255, 0), 2)

# 使用 matplotlib 显示结果
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.title('Detected Circles')
plt.axis('off')
plt.show()
