import numpy as np
import cv2
from skimage.transform import pyramid_gaussian

def sliding_window_core(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(int(image.shape[0] // 2), image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def sliding_window(img, verbous=False):
	# loop over the image pyramid
	winW, winH = (64,64)
	for (i, resized) in enumerate(pyramid_gaussian(img, downscale=2)):
		if resized.shape[0] < 64 or resized.shape[1] < 64:
			break
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window_core(resized, stepSize=64, windowSize=(winW, winH)):
	    	# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

	    	# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
	    	# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
	    	# WINDOW

			# since we do not have a classifier, we'll just draw the window
			clone = resized.copy()
			cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
			cv2.imshow("Window", clone)
			cv2.waitKey(50)
		cv2.waitKey(0)
	cv2.destroyAllWindows()