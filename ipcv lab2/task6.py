# import cv2
# import numpy as np
#
# image_folder = '/Users/lil/Desktop/ipcv lab2/img/landmark'
# image_list = []
#
# for i in range(1, 9):
#     image_path = f'{image_folder}/img{i}.jpg'
#     image = cv2.imread(image_path)
#     image_list.append(image)
# # 求八张图片的中位数
# alpha = 1.5 / len(image_list)
# median_image = cv2.addWeighted(image_list[0], alpha, image_list[1], alpha, 0)
#
# for i in range(2, len(image_list)):
#     median_image = cv2.addWeighted(median_image, 1.0, image_list[i], alpha, 0)
#
# # 可选：保存中位数图像
# cv2.imwrite('task6.jpg', median_image)

# 在循环结束后计算平均图像
# image_array = np.array(image_list)
# average_image = np.mean(image_array, axis=0).astype(np.uint8)
# window_size = 4
# height = average_image.shape[0]
# width = average_image.shape[1]
# output_image = np.zeros((height, width, 3), dtype=np.uint8)  # 3通道的彩色图像
# for x in range(height):
#     for y in range(width):
#         window = average_image[max(0, x - window_size // 2):min(height, x + window_size // 2 + 1),
#                max(0, y - window_size // 2):min(width, y + window_size // 2 + 1)]
#         output_image[x, y, :] = np.median(window, axis=(0, 1))
#
# # 保存彩色图像
# cv2.imwrite('task6_color.jpg', output_image)
import cv2
import numpy as np

image_folder = '/Users/lil/Desktop/ipcv lab2/img/landmark'
image_list = []

for i in range(1, 9):
    image_path = f'{image_folder}/img{i}.jpg'
    image = cv2.imread(image_path)
    image_list.append(image)

# 将图像列表转换为numpy数组
images_array = np.array(image_list)

# 计算中位数图像
median_image = np.median(images_array, axis=0).astype(np.uint8)

# 可选：保存中位数图像
cv2.imwrite('task6.jpg', median_image)



