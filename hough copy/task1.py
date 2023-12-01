import cv2
import numpy as np
import matplotlib.pyplot as plt

# 定义2D卷积函数
def convolution2d(image, kernel):
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    output = np.zeros_like(image, dtype=np.float32)

    # 填充图像
    image_padded = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # 执行2D卷积
    for x in range(image_height):
        for y in range(image_width):
            output[x, y] = (kernel * image_padded[x:x + kernel_height, y:y + kernel_width]).sum()

    return output

# 计算Sobel算子
def sobel(image):
    Kernelx = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])

    Kernely = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]])

    Ix = convolution2d(image, Kernelx)
    Iy = convolution2d(image, Kernely)

    gradient_magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    epsilon = 1e-5
    gradient_direction = np.arctan2(Iy, Ix + epsilon)
    gradient_direction = np.where(gradient_direction > np.pi, gradient_direction - 2 * np.pi, gradient_direction)
    gradient_direction = np.where(gradient_direction < -np.pi, gradient_direction + 2 * np.pi, gradient_direction)

    return Ix, Iy, gradient_magnitude, gradient_direction

# import cv2
# import numpy as np
#
# img = cv2.imread('coins1.png')
# # 将图像转换为灰度图像
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # 高斯滤波降噪
# gaussian_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
# # 利用Canny进行边缘检测
# edges_img = cv2.Canny(gaussian_img, 80, 180, apertureSize=3)
# # 自动检测圆
# circles1 = cv2.HoughCircles(edges_img, cv2.HOUGH_GRADIENT, 1, 100, param1=300, param2=10, minRadius=5, maxRadius=95)
#
# circles = circles1[0, :, :]
# circles = np.uint16(np.around(circles))
# for i in circles[:]:
#     cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
#
# cv2.imshow('task2.jpg', img)
# cv2.imwrite('task2.jpg',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
