import cv2
import numpy as np

# Load the mandrill image
input_image = cv2.imread('mandrill.jpg')

# Define a simple 3x3 kernel (you can adjust the values as needed)
kernel = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]]) / 9
output_image = np.zeros(input_image.shape)
height = input_image.shape[0]
width = input_image.shape[1]

# Remove the extra indentation from the following lines
for x in range(1, height - 1):
    for y in range(1, width - 1):
        image_box = input_image[x - 1:x + 2, y - 1:y + 2]
        output_image[x, y] = np.sum(image_box * kernel)

# Display the convolved image
cv2.imwrite('task1.jpg', output_image)

