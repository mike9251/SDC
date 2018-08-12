import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import sys
from calibrate_camera import undistort_image, calibrate_camera

class LaneDetector(object):
	def __init__(self, box_h = 50, box_w = 300, average_over = 5):
		self.box_h = box_h
		self.box_w = box_w
		self.average_over = average_over

		self.x_left = None
		self.y_left = None
		self.x_right = None
		self.y_right = None
		self.deque_left = []
		self.deque_right = []

		self.r = 0

		self.weights = np.array([0.1, 0.3, 0.5, 0.75, 1])

	def process_hist(self, image, offset):
		hist = np.sum(image[image.shape[0] - offset*self.box_h - self.box_h : image.shape[0] - offset*self.box_h,:], axis=0)
		mid = np.int(hist.shape[0] / 2)
		leftx = np.argmax(hist[:mid])
		rightx = np.argmax(hist[mid:]) + mid

		# For a new frame use x coordinates from a previous frame 
		if (self.x_left != None and self.x_right != None):
			#print("Use saved left ", self.x_left, "\nright ", self.x_right)
			self.deque_left = [int(np.median(self.x_left))]
			self.deque_right = [int(np.median(self.x_right))]
			self.x_left = None
			self.x_right = None

		if ((hist[rightx] == 255) or (hist[rightx] == 0)):
			if len(self.deque_right) > 0:
				rightx = int(np.mean(self.deque_right))

		if ((hist[leftx] == 255) or (hist[leftx] == 0)):
			if len(self.deque_left) > 0:
				leftx = int(np.mean(self.deque_left))

		if (len(self.deque_left) == self.average_over):
		    self.deque_left.pop(0)
		self.deque_left.append(leftx)

		if (len(self.deque_right) == self.average_over):
		    self.deque_right.pop(0)
		self.deque_right.append(rightx)

		# Calculate normalized coeffs
		weights = self.weights[len(self.weights) - len(self.deque_left) : ] / np.sum(self.weights[len(self.weights) - len(self.deque_left) : ])
		# Calculate centers for the left/right boxes wrt the previous
		leftx = int(np.dot(np.array([self.deque_left]), weights))
		rightx = int(np.dot(np.array([self.deque_right]), weights))

		return hist, mid, leftx, rightx

	def drawBox(self, image, offset, x):
		tlx = x - self.box_h*2
		tly = image.shape[0] - self.box_h - self.box_h * offset
		brx = x + self.box_h*2
		bry = tly + self.box_h
		cv2.rectangle(image, (tlx, tly), (brx, bry), [255, 0, 0], 2)
		return image

	def define_lines(self, image, draw_boxes = True):
		x_left = []
		y_left = []
		x_right = []
		y_right = []

		for it in range(image.shape[0] // self.box_h):
		    hist, mid, lx, rx = self.process_hist(image, it)

		    x_left.append(lx)
		    y_left.append(image.shape[0] - self.box_h//2 - it*self.box_h)

		    x_right.append(rx)
		    y_right.append(image.shape[0] - self.box_h//2 - it*self.box_h)

		    if draw_boxes:
		    	image = self.drawBox(image, it, int(lx))
		    	image = self.drawBox(image, it, int(rx))

		self.x_left = x_left
		self.x_right = x_right

		self.y_left = y_left
		self.y_right = y_right

		return image


	def get_lane(self, frame, roi):

		roi_with_boxes = self.define_lines(roi, draw_boxes = True)
		ploty = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
		# Calc coeffs
		coeffs_l = np.polyfit(self.y_left, self.x_left, 2)
		# Fit a line
		left_fitx = coeffs_l[0]*ploty**2 + coeffs_l[1]*ploty + coeffs_l[2]

		coeffs_r = np.polyfit(self.y_right, self.x_right, 2)
		right_fitx = coeffs_r[0]*ploty**2 + coeffs_r[1]*ploty + coeffs_r[2]

		left_line = np.array(np.transpose([np.vstack([left_fitx, ploty])]), np.int32)
		right_line = np.array(np.transpose([np.vstack([right_fitx, ploty])]), np.int32)

		# Calc radius of the curvature
		self.calc_curvature(left_fitx, ploty)

		self.calc_deviation(frame.shape[0], frame.shape[1])

		# Divide the lane into small regions
		rects_h = 25
		rects = np.array([[left_line[0], right_line[0], right_line[rects_h], left_line[rects_h]]])

		for i in range(rects_h, left_line.shape[0]-rects_h, rects_h):
		    top = np.array([[left_line[i], right_line[i]]])
		    bottom = np.array([[right_line[i+rects_h], left_line[i+rects_h]]])
		    rects = np.concatenate((rects, np.hstack([top, bottom])))
		    top = bottom

		lane_img = np.zeros_like(frame)

		# Go over all rects and draw them. Get the lane in warpPerspective transform form
		for (i, x) in enumerate(rects):
		    cv2.fillPoly(lane_img, [x], (0, 255, 0))

		return lane_img, roi_with_boxes

	def calc_curvature(self, x, y):
		ymax = np.max(y)

		ympp = 30/720
		xmpp = 3.7/700

		coeffs = np.polyfit(y * ympp, x * xmpp, 2)
		self.r = ((1 + (2 * coeffs[0] * ymax + coeffs[1]) ** 2) ** 1.5) / (2 * np.abs(coeffs[0]))

	def calc_deviation(self, h, w):
		xmpp = 3.7/700
		self.deviation = xmpp * (w/2) - xmpp * (self.x_right[0] + self.x_left[0])/2

	def GetRadius(self):
		return self.r

	def GetDeviation(self):
		return self.deviation

# Preprocess block
def extract_yellow_white(img):
	hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	low = np.uint8([15, 38, 115])
	hight = np.uint8([60, 204, 255])
	y_mask = cv2.inRange(hls_img, low, hight)
	y_img = cv2.bitwise_and(img, img, mask=y_mask)

	y_gray = cv2.cvtColor(y_img, cv2.COLOR_BGR2GRAY)
	_, by_img = cv2.threshold(y_gray, 127, 255, cv2.THRESH_BINARY)

	low = np.uint8([0, 200, 0])
	hight = np.uint8([180, 255, 255])
	w_mask = cv2.inRange(hls_img, low, hight)
	w_img = cv2.bitwise_and(img, img, mask=w_mask)
	w_gray = cv2.cvtColor(w_img, cv2.COLOR_BGR2GRAY)

	eq = cv2.equalizeHist(w_gray)
	_, eq = cv2.threshold(eq, 225, 255, cv2.THRESH_BINARY)
	beq = cv2.bitwise_or(eq, by_img)

	y_w_mask = cv2.bitwise_or(y_mask, w_mask)

	return cv2.bitwise_and(img, img, mask=y_w_mask), beq

def select_region(image):
    """
    It keeps the region surrounded by the `vertices` (i.e. polygon).  Other area is set to 0 (black).
    """
    # first, define the polygon by vertices
    rows, cols = image.shape[:2]
    bottom_left  = [cols*0.1, rows*0.95]
    top_left     = [cols*0.5, rows*0.6]
    bottom_right = [cols*0.9, rows*0.95]
    top_right    = [cols*0.6, rows*0.6] 
    # the vertices are an array of polygons (i.e array of arrays) and the data type must be integer
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)

    return vertices

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
    cv2.fillPoly(mask, np.array([vertices]), ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

# Transform Block
def four_point_transform(image):

    #print(image.shape)
    h, w = image.shape[:2]

    src = np.float32([
        [w, h-10],
        [0, h-10],
        [545, 460],
        [730, 460]])

    dst = np.float32([
        [w, h],
        [0, h],
        [0, 0],
        [w, 0]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    warped = cv2.warpPerspective(image, M, (w, h), flags=cv2.INTER_LINEAR)

    """cv2.circle(image, (w, h-10), 1, (0, 0, 255), thickness=10, lineType=8, shift=0)
    cv2.circle(image, (0, h-10), 1, (0, 0, 255), thickness=10, lineType=8, shift=0)
    cv2.circle(image, (545, 460), 1, (0, 0, 255), thickness=10, lineType=8, shift=0)
    cv2.circle(image, (730, 460), 1, (0, 0, 255), thickness=10, lineType=8, shift=0)

    cv2.circle(warped, (w, h), 1, (0, 0, 255), thickness=10, lineType=8, shift=0)
    cv2.circle(warped, (0, h), 1, (0, 0, 255), thickness=10, lineType=8, shift=0)
    cv2.circle(warped, (0, 0), 1, (0, 0, 255), thickness=10, lineType=8, shift=0)
    cv2.circle(warped, (w, 0), 1, (0, 0, 255), thickness=10, lineType=8, shift=0)

    image = cv2.resize(image, (640, 480))
    warped = cv2.resize(warped, (640, 480))

    cv2.namedWindow("dot_warped", cv2.WINDOW_NORMAL)
    cv2.imshow("dot_warped", warped)

    cv2.namedWindow("dot", cv2.WINDOW_NORMAL)
    cv2.imshow("dot", image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()"""

    return warped, M, Minv

def pipeline(image, verbouse=False):
	yw_img, binary = extract_yellow_white(cv2.GaussianBlur(image, (9, 9), 0))

	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	blured_img = cv2.GaussianBlur(gray_img, (9, 9), 0)

	sobel_x = cv2.Sobel(blured_img, cv2.CV_64F, 1, 0, ksize=5)
	sobel_y = cv2.Sobel(blured_img, cv2.CV_64F, 0, 1, ksize=5)

	sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
	sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

	_, sobel_mag = cv2.threshold(sobel_mag, 120, 255, cv2.THRESH_BINARY)

	binary = cv2.bitwise_or(binary, sobel_mag)

	vertices = select_region(binary)

	roi_img_sobel = region_of_interest(binary, vertices)

	kernel = np.ones((9, 9), np.uint8)
	closing = cv2.morphologyEx(roi_img_sobel.astype(np.uint8), cv2.MORPH_CLOSE, kernel)

	closing, M, Minv = four_point_transform(closing)

	return closing, M, Minv


def get_args(name='default', video_file="None"):
	return video_file

filename = get_args(*sys.argv)

def process_video(video_file):
	#out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280, 720))
	fake_rgb = np.zeros((240,320,3), dtype=np.uint8)
	cap = cv2.VideoCapture(video_file)
	# calibrate the camera
	mtx, dist, rvecs, tvecs = calibrate_camera('camera_cal/', False)
	lane = LaneDetector()
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		# undistort the image
		frame = cv2.undistort(frame, mtx, dist, None, mtx)
		roi, M, Minv = pipeline(frame, True)

		lane_img, roi_with_boxes = lane.get_lane(frame, roi)
		roi_with_boxes = cv2.resize(roi_with_boxes, (320, 240))

		# Inverse transform of the ROI with the lane and blend it with the original image
		lane_unwarped = cv2.warpPerspective(lane_img, Minv, (frame.shape[1], frame.shape[0]))
		result = cv2.addWeighted(frame, 1., lane_unwarped, 0.7, 0)

		r = lane.GetRadius()
		font = cv2.FONT_HERSHEY_SIMPLEX
		text = "Radius of Curvature: {} m".format(int(r))
		cv2.putText(result, text,(10,50), font, 1, (255,255,255), 2)

		deviation = lane.GetDeviation()
		font = cv2.FONT_HERSHEY_SIMPLEX
		text = "Deviation from the center of the lane: %.3f m" % abs(deviation)
		cv2.putText(result, text,(10,100), font, 1, (255,255,255), 2)

		fake_rgb[:, :, 0] = roi_with_boxes
		fake_rgb[:, :, 1] = roi_with_boxes
		fake_rgb[:, :, 2] = roi_with_boxes

		result[0:240, 960:, :] = fake_rgb

		cv2.namedWindow("result", cv2.WINDOW_NORMAL)
		cv2.imshow("result", result)

		#out.write(result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

process_video(filename)