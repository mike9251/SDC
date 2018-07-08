import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob

def calibrate_camera(image_dir, verbouse=False):
	image_names = glob.glob('camera_cal/*.jpg')

	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((9*6, 3), np.float32)
	objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)

	objpoints = []  # 3d points in real world space
	imgpoints = []  # 2d points in image plane.

	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

	for name in image_names:
	    img_bgr = cv2.imread(name)
	    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

	    # Find the chess board corners
	    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

	        # If found, add object points, image points (after refining them)
	    if ret == True:
	        objpoints.append(objp)

	        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
	        imgpoints.append(corners2)

	        # Draw and display the corners
	        if verbouse == True:
	            img = cv2.drawChessboardCorners(img_bgr, (9,6), corners, ret)
	            cv2.imshow('img', img)
	            cv2.waitKey(0)

	            cv2.destroyAllWindows()

	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
	return mtx, dist, rvecs, tvecs

def undistort_image(image, verbouse=False):
	mtx, dist, rvecs, tvecs = calibrate_camera('camera_cal/', False)
	undistort_img = cv2.undistort(image, mtx, dist, None, mtx)

	if verbouse == True:
		#fig, ax = plt.subplots(nrows=1, ncols=2)
		#ax[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
		#ax[1].imshow(cv2.cvtColor(undistort_img, cv2.COLOR_BGR2RGB))
		#plt.show()
		cv2.imshow("input", image)
		cv2.imshow("output", undistort_img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

	return undistort_img


image_names = glob.glob('camera_cal/*.jpg')

image = cv2.imread('camera_cal/calibration2.jpg')

undist_img = undistort_image(image, True)