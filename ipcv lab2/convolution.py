################################################
#
# COMS30068 - convolution.py
# TOPIC: create, save and display an image
#
# Getting-Started-File for OpenCV
# University of Bristol
#
################################################

import numpy as np
import cv2
import os
import sys
import argparse
import matplotlib.pyplot as plt

    # 均值滤波器是一种低通滤波器，它将图像中每个像素的值替换为其周围像素的平均值。这种滤波器通常用于图像平滑，它有助于去除图像中的高频噪声和细节，使图像变得更加模糊。
    #
    # 输出图像显示了一个经过平滑处理的版本，较原始图像具有更加模糊的外观。这种模糊可以降低图像的细节水平，适用于某些应用，如去噪或模糊背景以突出前景。
    #
    # 为什么需要因子
    # 1 / 9:
    #
    # 均值滤波器中的因子
    # 1 / 9
    # 是为了进行归一化。在卷积操作中，每个卷积核的权重用于计算输出图像中每个像素的值。如果没有归一化，输出图像的亮度可能会显著增加，因为它是通过将大量像素值相加来计算的。
    #
    # 归一化确保输出图像的亮度与原始图像大致相同，因此输出图像在视觉上仍然具有合理的亮度级别。这是图像处理中的一个重要考虑因素，以确保输出图像看起来自然。
    #
    # 总之，均值滤波器的因子
    # 1 / 9
    # 是为了确保输出图像在平滑操作后保持合理的亮度级别，并且避免亮度的不合理增加。这是图像处理中常用的一种技巧，尤其是在平滑和滤波操作中。
















    # LOADING THE IMAGE
# Example usage: python convolution.py -n car1.png
parser = argparse.ArgumentParser(description='Convert RGB to GRAY')
parser.add_argument('-name', '-n', type=str, default='car1.png')
args = parser.parse_args()

# ==================================================
def GaussianBlur(input, size):

	# intialise the output using the input
	blurredOutput = np.zeros([input.shape[0], input.shape[1]], dtype=np.float32)
	# create the Gaussian kernel in 1D
	kX = cv2.getGaussianKernel(size,1)
	kY = cv2.getGaussianKernel(size,1)
	# make it 2D multiply one by the transpose of the other
	kernel = kX * kY.T

	# CREATING A DIFFERENT IMAGE kernel WILL BE NEEDED
	# TO PERFORM OPERATIONS OTHER THAN GUASSIAN BLUR!!!

	# we need to create a padded version of the input
	# or there will be border effects
	kernelRadiusX = round(( kernel.shape[0] - 1 ) / 2)
	kernelRadiusY = round(( kernel.shape[1] - 1 ) / 2)

	paddedInput = cv2.copyMakeBorder(input,
		kernelRadiusX, kernelRadiusX, kernelRadiusY, kernelRadiusY,
		cv2.BORDER_REPLICATE)

	# now we can do the convoltion
	for i in range(0, input.shape[0]):
		for j in range(0, input.shape[1]):
			patch = paddedInput[i:i+kernel.shape[0], j:j+kernel.shape[1]]
			sum = (np.multiply(patch, kernel)).sum()
			# if you want to try to go through all pixels like cpp
			#sum = 0.0;
			#for m in range(-kernelRadiusX, kernelRadiusX+1):
			#	for n in range(-kernelRadiusY, kernelRadiusY+1):
			#		# find the correct indices we are using
			#		imagex = i + m + kernelRadiusX
			#		imagey = j + n + kernelRadiusY
			#		kernelx = m + kernelRadiusX
			#		kernely = n + kernelRadiusY
			#		# get the values from the padded image and the kernel
			#		imageval = paddedInput[imagex, imagey]
			#		kernalval = kernel[kernelx, kernely]
			#		# do the multiplication
			#		sum += imageval * kernalval
			# set the output value as the sum of the convolution
			blurredOutput[i, j] = sum

	return blurredOutput

# ==== MAIN ==============================================
imageName = args.name

# ignore if no such file is present.
if not os.path.isfile(imageName):
    print('No such file')
    sys.exit(1)

# Read image from file
image = cv2.imread(imageName, 1)

# ignore if image is not array.
if not (type(image) is np.ndarray):
    print('Not image data')
    sys.exit(1)

# CONVERT COLOUR, BLUR AND SAVE
gray_image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY );
gray_image = gray_image.astype(np.float32)

# apply Gaussian blur
carBlurred = GaussianBlur(gray_image,23);
# save image
cv2.imwrite( "blur.jpg", carBlurred );





