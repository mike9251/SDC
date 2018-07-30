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

		#self.beta = 0.5
		#self.avgx_left = 0
		#self.avgx_right = 0
		self.x_left = []
		self.y_left = []
		self.x_right = []
		self.y_right = []
		self.deque_left = []
		self.deque_right = []
		self.nframe = 0

		self.r = 0

		self.weights = np.array([0.1, 0.3, 0.5, 0.75, 1])

	def process_hist(self, image, offset):
		hist = np.sum(image[image.shape[0] - offset*self.box_h - self.box_h : image.shape[0] - offset*self.box_h,:], axis=0)
		mid = np.int(hist.shape[0] / 2)
		leftx = np.argmax(hist[:mid])
		rightx = np.argmax(hist[mid:]) + mid
		#print("rx = ", rightx, "hist[rx] = ", hist[rightx])

		if (self.x_left and self.x_right):
			#print("Use saved left ", self.x_left, "\nright ", self.x_right)
			self.deque_left = self.x_left#[int(np.median(self.x_left))]
			self.deque_right = self.x_right#[int(np.median(self.x_right))]
			self.x_left = []
			self.x_right = []

		if ((hist[rightx] == 255) or (hist[rightx] == 0)):
			if len(self.deque_right) > 0:
				rightx = int(np.mean(self.deque_right))
				#print("Recalculate rx = ", rightx, "hist[rx] = ", hist[rightx])

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

		#print("Frame # ", self.nframe)

		for it in range(image.shape[0] // self.box_h):
		    hist, mid, lx, rx = self.process_hist(image, it)

		    x_left.append(lx)
		    y_left.append(image.shape[0] - self.box_h//2 - it*self.box_h)

		    x_right.append(rx)
		    y_right.append(image.shape[0] - self.box_h//2 - it*self.box_h)

		    if draw_boxes:
		    	image = self.drawBox(image, it, int(lx))
		    	image = self.drawBox(image, it, int(rx))

		self.x_left = [int(np.median(x_left))]
		self.x_right = [int(np.median(x_right))]

		self.x = x_right
		self.y = y_right

		#self.nframe += 1

		#print("Save left ", self.x_left, "\nrigth ", self.x_right)

		return image, x_left, y_left, x_right, y_right

	def get_lane(self, frame, roi):

		roi_with_boxes, xl_coords, yl_coords, xr_coords, yr_coords = self.define_lines(roi, draw_boxes = True)
		ploty = np.linspace(0, frame.shape[0] - 1, frame.shape[0])
		# Calc coeffs
		coeffs_l = np.polyfit(yl_coords, xl_coords, 2)
		# Fit a line
		left_fitx = coeffs_l[0]*ploty**2 + coeffs_l[1]*ploty + coeffs_l[2]

		coeffs_r = np.polyfit(yr_coords, xr_coords, 2)
		right_fitx = coeffs_r[0]*ploty**2 + coeffs_r[1]*ploty + coeffs_r[2]

		left_line = np.array(np.transpose([np.vstack([left_fitx, ploty])]), np.int32)
		right_line = np.array(np.transpose([np.vstack([right_fitx, ploty])]), np.int32)

		# Calc radius of the curvature
		avg_fit_x = np.array(right_fitx) - np.array(left_fitx)

		print("X.shape = ", avg_fit_x.shape, "Y.shape = ", np.array(ploty).shape)

		self.r = self.calc_curvature(avg_fit_x, np.array(ploty))



		# just 4 points
		#lane2 = np.array([left_line[0], right_line[0], right_line[-1], left_line[-1]])
		#print("\nLane2 shape = \n", lane2.shape, lane2)
		#cv2.fillPoly(out, [lane2], (0, 255, 0))

		# Divide the lane into small regions
		rects_h = 25
		rects = np.array([[left_line[0], right_line[0], right_line[rects_h], left_line[rects_h]]])

		#print("\nRect shape = \n", rects.shape)
		for i in range(rects_h, left_line.shape[0]-rects_h, rects_h):
		    top = np.array([[left_line[i], right_line[i]]])
		    bottom = np.array([[right_line[i+rects_h], left_line[i+rects_h]]])
		    rects = np.concatenate((rects, np.hstack([top, bottom])))
		    top = bottom

		#print("\nVerts shape = \n", rects.shape)

		#lane3 = np.array([x for x in rects])
		lane_img = np.zeros_like(frame)

		# Go over all rects and draw them. Get the lane in warpPerspective transform form
		for (i, x) in enumerate(rects):
		    cv2.fillPoly(lane_img, [x], (0, 255, 0))

		#cv2.namedWindow("lane_img", cv2.WINDOW_NORMAL)
		#cv2.imshow("lane_img", lane_img)

		# Transform the original image and blend it with the lane
		#warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
		#warped = cv2.addWeighted(warped, 1, lane_img, 0.7, 0)
		#cv2.namedWindow("img", cv2.WINDOW_NORMAL)
		#cv2.imshow("img", warped)

		# Inverse warping of the ROI
		#unwarped = cv2.warpPerspective(warped, Minv, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
		#cv2.namedWindow("img4", cv2.WINDOW_NORMAL)
		#cv2.imshow("img4", unwarped)

		return lane_img

	def calc_curvature(self, x, y):
		y = np.max(y)

		ympp = 30/720
		xmpp = 3.7/700

		coeffs = np.polyfit(y * ympp, x * xmpp, 2)
		self.r = ((1 + (2 * coeffs[0] * y + coeffs[1]) ** 2) ** 1.5) / (2 * np.abs(coeffs[0]))

		return self.r

# Preprocess block
def extract_yellow_white(img):
	hls_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
	low = np.uint8([15, 38, 115])
	hight = np.uint8([60, 204, 255])
	y_mask = cv2.inRange(hls_img, low, hight)
	y_img = cv2.bitwise_and(img, img, mask=y_mask)

	y_gray = cv2.cvtColor(y_img, cv2.COLOR_BGR2GRAY)
	_, by_img = cv2.threshold(y_gray, 127, 255, cv2.THRESH_BINARY)
	#print(by_img.shape)

	#cv2.imshow("Y", cv2.bitwise_and(img, img, mask=y_mask))
	#cv2.waitKey(0)
	#cv2.imshow("Y_GRAY", y_gray)
	#cv2.waitKey(0)
	#cv2.imshow("Y_GRAY_th", by_img)
	#cv2.waitKey(0)

	low = np.uint8([0, 200, 0])
	hight = np.uint8([180, 255, 255])
	w_mask = cv2.inRange(hls_img, low, hight)
	w_img = cv2.bitwise_and(img, img, mask=w_mask)
	w_gray = cv2.cvtColor(w_img, cv2.COLOR_BGR2GRAY)

	eq = cv2.equalizeHist(w_gray)
	#cv2.imshow("eq", eq)
	#cv2.waitKey(0)
	_, eq = cv2.threshold(eq, 225, 255, cv2.THRESH_BINARY)
	#cv2.imshow("W_GRAY_th", eq)
	#cv2.waitKey(0)
	beq = cv2.bitwise_or(eq, by_img)
	#print(eq.shape)

	y_w_mask = cv2.bitwise_or(y_mask, w_mask)

	#cv2.imshow("YW", cv2.bitwise_and(img, img, mask=y_w_mask))
	#cv2.waitKey(0)
	return cv2.bitwise_and(img, img, mask=y_mask), beq

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

    ordered_pts = np.zeros((4, 2), dtype="float32")
    ordered_pts[0] =  (cols*0.4, rows*0.6)
    ordered_pts[1] = (cols*0.6, rows*0.6)
    ordered_pts[2] = (cols*0.85, rows*0.95)
    ordered_pts[3] = (cols*0.15, rows*0.95)

    return vertices, ordered_pts

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
    #plt.imshow(mask)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def order_pointa(pts):
    print(pts)
    rect = np.zeros((4, 2), dtype = "float32")

    s = pts.sum(axis=1)
    print("Sum:", s)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    print("Diff: ", diff)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

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

    return warped, M, Minv

def pipeline(image, verbouse=False):
	#image = undistort_image(image)
	yw_img, binary = extract_yellow_white(cv2.GaussianBlur(image, (9, 9), 0))

	gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#cv2.imshow("grfay", gray_img)
	#cv2.waitKey(0)

	blured_img = cv2.GaussianBlur(gray_img, (9, 9), 0)

	sobel_x = cv2.Sobel(blured_img, cv2.CV_64F, 1, 0, ksize=5)
	sobel_y = cv2.Sobel(blured_img, cv2.CV_64F, 0, 1, ksize=5)

	sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
	sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

	_, sobel_mag = cv2.threshold(sobel_mag, 120, 255, cv2.THRESH_BINARY)

	#cv2.imshow("thresh", sobel_mag)
	#cv2.waitKey(0)

	binary = cv2.bitwise_or(binary, sobel_mag)

	vertices, vert_arr = select_region(binary)
	#print("Coord: ", order_pointa(vert_arr))
	roi_img_sobel = region_of_interest(binary, vertices)
	#cv2.imshow("bin", binary)
	#cv2.waitKey(0)

	kernel = np.ones((9, 9), np.uint8)
	closing = cv2.morphologyEx(roi_img_sobel.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
	#cv2.imshow("closing", closing)
	#cv2.waitKey(0)

	#img2 = np.copy(closing)
	#color = [255, 0, 0]
	#thikness = 2
	#cv2.line(img2, (vert_arr[0][0], vert_arr[0][1]), (vert_arr[3][0], vert_arr[3][1]), color, thikness)
	#cv2.line(img2, (vert_arr[1][0], vert_arr[1][1]), (vert_arr[2][0], vert_arr[2][1]), color, thikness)


	closing, M, Minv = four_point_transform(closing)
	#print("closing.shape = ", closing.shape)

	return closing, M, Minv


def get_args(name='default', video_file="None"):
	return video_file

filename = get_args(*sys.argv)

"""def process_video(video_file):
	#out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280, 720))
	cap = cv2.VideoCapture(video_file)
	cv2.namedWindow("frame")
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		img = pipeline(frame)
		#out.write(img)
		cv2.imshow("frame", img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()"""

def process_video(video_file):
	#out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280, 720))
	cap = cv2.VideoCapture(video_file)
	# calibrate the camera
	#mtx, dist, rvecs, tvecs = calibrate_camera('camera_cal/', False)
	lane = LaneDetector()
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		# undistort the image
		#frame = cv2.undistort(frame, mtx, dist, None, mtx)
		roi, M, Minv = pipeline(frame, True)

		lane_img = lane.get_lane(frame, roi)
		# Inverse transform of the ROI with the lane and blend it with the original image
		lane_unwarped = cv2.warpPerspective(lane_img, Minv, (frame.shape[1], frame.shape[0]))
		result = cv2.addWeighted(frame, 1., lane_unwarped, 0.7, 0)

		r = lane.r#lane.calc_curvature()
		font = cv2.FONT_HERSHEY_SIMPLEX
		text = "Radius of Curvature: {} m".format(int(r))
		cv2.putText(result, text,(10,50), font, 1, (255,255,255), 2)

		cv2.namedWindow("result", cv2.WINDOW_NORMAL)
		cv2.imshow("result", result)
		#out.write(img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

process_video(filename)