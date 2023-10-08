# import cv2
# import cv2 as cv
# import numpy as np
#
# image = cv.imread('mandrillRGB.jpg',1)
# thresh = 200
# for y in range(0, image.shape[0]):  # go through all rows (or scanlines)
# 	for x in range(0, image.shape[1]):  # go through all columns
# 		pixelBlue = image[y, x, 0]
# 		pixelGreen = image[y, x, 1]
# 		pixelRed = image[y, x, 2]
# 		if (pixelBlue>200):
# 			image[y, x, 0] = 255
# 			image[y, x, 1] = 255
# 			image[y, x, 2] = 255
# 		else:
# 			image[y, x, 0] = 0
# 			image[y, x, 1] = 0
# 			image[y, x, 2] = 0
#
# # Save thresholded image
# cv2.imwrite("colourthr.jpg", image)
# --------以上是对彩色图像进行阈值操作-------
