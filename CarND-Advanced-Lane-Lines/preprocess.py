import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import sys

def extract_yellow_white(img):
    hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    low = np.uint8([15, 38, 115])
    hight = np.uint8([35, 204, 255])
    y_mask = cv2.inRange(hls_img, low, hight)
    low = np.uint8([0, 200, 0])
    hight = np.uint8([180, 255, 255])
    w_mask = cv2.inRange(hls_img, low, hight)
    y_w_mask = cv2.bitwise_or(y_mask, w_mask)

    return cv2.bitwise_and(img, img, mask=y_w_mask)

def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.4, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return region_of_interest(image, vertices)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, np.array([vertices], dtype=np.int32), ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    #plt.imshow(mask)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def preprocess(image, verbouse=False):

	yw_img = extract_yellow_white(image)

	gray_img = cv2.cvtColor(yw_img, cv2.COLOR_BGR2GRAY)
	blured_img = cv2.GaussianBlur(gray_img, (7, 7), 0)

	sobel_x = cv2.Sobel(blured_img, cv2.CV_64F, 1, 0, ksize=9)
	sobel_y = cv2.Sobel(blured_img, cv2.CV_64F, 0, 1, ksize=9)

	sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
	sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

	_, sobel_mag = cv2.threshold(sobel_mag, 100, 255, cv2.THRESH_BINARY)

	roi_img_sobel = select_region(sobel_mag)

	kernel = np.ones((9, 9), np.uint8)
	closing = cv2.morphologyEx(roi_img_sobel.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

	return closing


image_names = glob.glob('test_images/*.jpg')

image = cv2.imread('test_images/test3.jpg')

prep_img = preprocess(image, True)

cv2.imshow("img", prep_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



def get_args(name='default', video_file="None"):
	return video_file

filename = get_args(*sys.argv)

def process_video(video_file):
	#out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280, 720))
	cap = cv2.VideoCapture(video_file)
	cv2.namedWindow("frame")
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		img = preprocess(frame)
		#out.write(img)
		cv2.imshow("frame", img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

process_video(filename)