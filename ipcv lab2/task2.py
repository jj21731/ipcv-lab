import cv2
import numpy as np

# Load the mandrill image
input_image = cv2.imread('car1.png',cv2.IMREAD_GRAYSCALE)

# Define a simple 3x3 kernel (you can adjust the values as needed)
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]) / 9
blur_image = np.zeros(input_image.shape)
height = input_image.shape[0]
width = input_image.shape[1]

# Remove the extra indentation from the following lines
for x in range(1, height - 1):
    for y in range(1, width - 1):
        image_box = input_image[x - 1:x + 2, y - 1:y + 2]
        blur_image[x, y] = np.sum(image_box * kernel)
        sharpen_image = input_image + 30 * (input_image - blur_image)
        sharpen_image = np.clip(sharpen_image, 0, 255).astype(np.uint8)
cv2.imshow('Sharpened Image', sharpen_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Save the sharpened image
cv2.imwrite('task2.jpg', sharpen_image)

