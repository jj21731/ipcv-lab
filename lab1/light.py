# import cv2
#
# # Load the dark image
# dark_image = cv2.imread('darkBristol.png')
#
# if dark_image is None:
#     print("Error: Unable to load the image!")
# else:
#     # Enhance brightness
#     bright_image = cv2.convertScaleAbs(dark_image, alpha=5.1, beta=5)
#
#     # Save the enhanced image
#     cv2.imwrite('brightBristol.png', bright_image)
#
#     # Display the original and enhanced images
#     cv2.imshow('Original Image', dark_image)
#     cv2.imshow('Enhanced Image', bright_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
import cv2
import numpy as np

image = cv2.imread("darkBristol.png", 1)
 
# enhance brightness
image = np.power(image, 1.5)

cv2.imwrite("enhancedBristol.jpg", image)
