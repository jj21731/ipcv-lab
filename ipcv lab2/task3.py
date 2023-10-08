# import cv2
# image = cv2.imread('car2.png')
# median_filtered = cv2.medianBlur(image,5)
# cv2.imwrite('task3.jpg',median_filtered)
# ---------cv2---------
import numpy as np
import cv2


image = cv2.imread('car2.png',cv2.IMREAD_GRAYSCALE)
window_size = 5
height = image.shape[0]
width = image.shape[1]

output_image = np.zeros((height,width), dtype = np.uint8)

for x in range(height):
    for y in range(width):
     window = image[max(0,x - window_size//2):min(height,x + window_size//2+1),
               max(0,y - window_size//2):min(width,y + window_size//2+1)]
     output_image[x,y] = np.median(window)
cv2.imwrite('task3.jpg',output_image)