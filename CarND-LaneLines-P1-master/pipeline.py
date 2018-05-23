import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
#from collections import deque
import cv2
import sys

# exponential moving average params
t = 0
beta = 0.98
avg_right_line = 0
avg_left_line = 0

stack_r = []

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
    plt.imshow(mask)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

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

def average_lines(lines):
    left_line = []
    left_line_length = []
    right_line = []
    right_line_length = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            bias = y2 - slope * x2
            length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)

            if slope >= 0:
                right_line.append((slope, bias))
                right_line_length.append(length)
            else:
                left_line.append((slope, bias))
                left_line_length.append(length)
    left_lane = np.dot(left_line_length, left_line) / np.sum(left_line_length) if len(left_line_length) > 0 else None
    right_lane = np.dot(right_line_length, right_line) / np.sum(right_line_length) if len(right_line_length) > 0 else None
    
    return left_lane, right_lane

def make_line_points(y1, y2, line):
    slope, bias = line
    
    x1 = np.abs(int((bias - y1)/slope))
    x2 = np.abs(int((bias - y2)/slope))
    y1 = int(y1)
    y2 = int(y2)
    
    return ((x1, y1), (x2, y2))

def lane_lines(image, lines):
    left_lane, right_lane = average_lines(lines)
    global t
    global beta
    global avg_right_line
    global avg_left_line
    global stack_r
    avg_l = 0
    avg_r = 0

    if len(stack_r) >= 30:
    	print("Pop an element from stack")
    	stack_r.pop()

    stack_r.append((left_lane, right_lane))

    for left, right in stack_r:
    	avg_l = avg_l + left
    	avg_r = avg_r + right
    avg_l = avg_l / len(stack_r)
    avg_r = avg_r / len(stack_r)
    avg_left_line = avg_l
    avg_right_line = avg_r


    """if t == 0:
    	avg_left_line = left_lane
    	avg_right_line = right_lane
    	t = t + 1
    elif t >= 1:
        avg_left_line = beta * avg_left_line + (1 - beta) * left_lane
        avg_right_line = beta * avg_right_line + (1 - beta) * right_lane
        left_lane = avg_left_line
        right_lane = avg_right_line
        t = t + 1"""

    y1 = image.shape[0]
    y2 = int(y1 * 0.6)
    
    left_line = make_line_points(y1, y2, left_lane)
    right_line = make_line_points(y1, y2, right_lane)

    return left_line, right_line

def draw_lane_lines(image, lines, color=[255, 0, 0], thickness=20):
    # make a separate image to draw lines and combine with the orignal later
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line,  color, thickness)
    # image1 * α + image2 * β + λ
    # image1 and image2 must be the same shape.
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

def pipeline(img):

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blured_img = cv2.GaussianBlur(gray_img, (7, 7), 0)
    
    edges = cv2.Canny(blured_img, 150, 200)
    
    roi_img = select_region(edges)

    lines_per_image = cv2.HoughLinesP(roi_img, 2, np.pi/180, 35, np.array([]), minLineLength=10, maxLineGap=100)
    
    lane_image = draw_lane_lines(img, lane_lines(img, lines_per_image))
    
    return lane_image


def get_args(name='default', video_file="None"):
	return video_file

filename = get_args(*sys.argv)

def process_video(video_file):

	cap = cv2.VideoCapture(video_file)
	cv2.namedWindow("frame")
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		img = pipeline(frame)
		cv2.imshow("frame", img)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

process_video(filename)