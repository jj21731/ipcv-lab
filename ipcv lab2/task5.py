import cv2
import numpy as np

image_folder = '/Users/lil/Desktop/ipcv lab2/img/waterfall'
image_list = []

for i in range(1, 31):
    image_path = f'{image_folder}/{i:05d}.png'  # f'是格式化字符串，保证了图像文件的完整性，i:05表示05d表示将i格式化为一个至少5位数的十进制整数，不足的位数用0填充。例如，如果i的值为3，那么这个表达式将生成'00003'。
    image = cv2.imread(image_path)
    image_list.append(image)

# 在循环结束后计算平均图像
image_array = np.array(image_list)
average_image = np.mean(image_array, axis=0).astype(np.uint8)
cv2.imwrite('task5.jpg', average_image)
# task 5思想是为了制作具有平滑效果的瀑布图像，您可以将这30帧瀑布剪辑叠加在一起，然后进行平均处理。这将有助于减少过度曝光，因为它会减弱每个图像的亮度，并突出水流动的效