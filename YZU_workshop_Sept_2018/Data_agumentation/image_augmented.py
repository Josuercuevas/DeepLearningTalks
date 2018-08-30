'''
    This file is part of the September 2018 Workshop at Yuan Ze University.

    You can use these examples in the way you seem fit, though I can't make sure
    it will work fine in your case.
'''

# opencv module
import cv2 as cv

# pillow module
from PIL import ImageEnhance, Image

# numpy module
import numpy as np

image2use = "test_image.jpg"

# loading image
img_read = cv.imread(image2use)

# just to make it larger
img_read = cv.resize(img_read, (1280, 720), 1)


# show the image
cv.imshow("Image Loaded", img_read)
cv.waitKey()

# Here we will do some changes

# brightness
pil_image = Image.fromarray(img_read, "RGB")
brighter = ImageEnhance.Brightness(pil_image)

for idx in range(3):
	factor = 0.4 * (idx+1)
	bright_image = brighter.enhance(factor)
	brightened = np.asarray(bright_image)
	cv.imshow("Image Change Brightness: %f"%factor, brightened)
	cv.waitKey()

# Color
pil_image = Image.fromarray(img_read, "RGB")
colorer = ImageEnhance.Color(pil_image)

for idx in range(3):
	factor = 0.3 * (idx+1)
	colored_image = colorer.enhance(factor)
	colored = np.asarray(colored_image)
	cv.imshow("Image changed color: %f"%factor, colored)
	cv.waitKey()

# Blurring
pil_image = Image.fromarray(img_read, "RGB")
blurrer = ImageEnhance.Sharpness(pil_image)

for idx in range(3):
	factor = idx
	blurred_image = blurrer.enhance(factor)
	blurred = np.asarray(blurred_image)
	cv.imshow("Image Blurred: %f"%factor, blurred)
	cv.waitKey()

# noise
for idx in range(3):
	factor = 0.3*idx
	noised = np.array(img_read).astype(np.float)*(1-factor) + \
			np.array(np.random.randint(0, 255, img_read.shape)).astype(np.float)*factor
	noised = np.array(noised).astype(np.uint8)
	cv.imshow("Image with Noise: %f" % factor, noised)
	cv.waitKey()

# image flipping Y-axis
pil_image = Image.fromarray(img_read, "RGB")
flipped_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
flippedY = np.asarray(flipped_image)
cv.imshow("Flip Y axis", flippedY)
cv.waitKey()

# image flipping X-axis
pil_image = Image.fromarray(img_read, "RGB")
flipped_image = pil_image.transpose(Image.FLIP_TOP_BOTTOM)
flippedX = np.asarray(flipped_image)
cv.imshow("Flip Y axis", flippedX)
cv.waitKey()


# now how do we send it to tensorflow?
import tensorflow as tf
# TFimage = tf.placeholder("uint8", shape=[1280, 720, 3]) # will crash why?
TFimage = tf.placeholder("uint8", shape=[720, 1280, 3])

with tf.Session() as sess:
	# eval expressions with parameters for the image
	image_got = sess.run(TFimage, feed_dict={TFimage: flippedX})
	cv.imshow("Image got from TF", image_got)
	cv.waitKey()

