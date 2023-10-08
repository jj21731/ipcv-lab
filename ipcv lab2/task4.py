import cv2
import deconvolution



image = cv2.imread('car3.png',cv2.IMREAD_GRAYSCALE)
image_length = 15
image_angle = 3
signal_to_noise_ratio = 0.001
regularization_parameter = 0
recover = deconvolution.WienerDeconvoluition(image,30,1,0.001,0)
cv2.imwrite('task4.jpg', recover)