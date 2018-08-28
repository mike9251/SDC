from load_data import load_data, color_convert
from svm import load_model, train_svm
from descriptor import get_descriptor, get_hist_feature, get_spatial_bin_feature, get_hog_features, get_descriptor_for_train
#from processing import sliding_window
import cv2
import scipy
import numpy as np
import sys
from skimage.transform import pyramid_gaussian
from skimage import img_as_ubyte
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def normalize_image(img):
    """
    Normalize image between 0 and 255 and cast to uint8
    (useful for visualization)
    """
    img = np.float32(img)

    img = img / img.max() * 255

    return np.uint8(img)

def heat_map(img, windows, thresh, verbose=False, nms=False):

	h, w, c = img.shape

	heat_map = np.zeros((h, w), dtype=np.uint8)

	for window in windows:
		if nms==True:
			x1, y1, x2, y2 = window
		else:
			x1, y1 = window[0]
			x2, y2 = window[1]
		heat_map[y1:y2, x1:x2] += 1

	_, heat_map_thresh = cv2.threshold(heat_map, thresh, 255, type=cv2.THRESH_BINARY)
	heat_map_thresh = cv2.morphologyEx(heat_map_thresh, op=cv2.MORPH_CLOSE, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
	if verbose:
		f, ax = plt.subplots(1, 3)
		ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
		ax[1].imshow(heat_map, cmap='hot')
		ax[2].imshow(heat_map_thresh, cmap='hot')
		plt.show()

	return heat_map, heat_map_thresh


def draw_segments(img, labeled_img, num_objects):
	"""
	Starting from labeled regions, draw enclosing rectangles in the original color frame.
	"""
	# Iterate through all detected cars
	boxes = []
	for box in range(1, num_objects+1):

		# Find pixels with each box label value
		rows, cols = np.where(labeled_img == box)

		# Find minimum enclosing rectangle
		x1, y1 = np.min(cols), np.min(rows)
		x2, y2 = np.max(cols), np.max(rows)

		cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=3)
		boxes.append([(x1, y1), (x2, y2)])

	return img, boxes

def NMS(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
 
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
 
	# initialize the list of picked indexes	
	pick = []
 
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]

	print("x1 shape = ", x1.shape)
 
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	print("area shape = ", area.shape)
	idxs = np.argsort(y2)
 
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
 
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
 
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
 
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
 
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick].astype("int")


def detect_cars(img, scale, clf, feature_scaler):
	boxes = []
	y_start = 400
	y_end = 600

	resize_w = resize_h = 64
	window = 64
	pix_per_cell = 8

	roi = color_convert(img[y_start:y_end, :, :], 'YCrCb')

	if(scale != 1):
		roi = cv2.resize(roi, (np.int(roi.shape[0] // scale), np.int(roi.shape[1] // scale)))

	H, W, C = roi.shape

	n_cell_x = (W // pix_per_cell) - 1
	n_cell_y = (H // pix_per_cell) - 1

	n_cell_per_window = (window // pix_per_cell) - 1

	n_cell_per_step = 4

	n_step_in_cells_x = (n_cell_x - n_cell_per_window) // n_cell_per_step
	n_step_in_cells_y = (n_cell_y - n_cell_per_window) // n_cell_per_step

	hog_c1 = get_hog_features(roi[:,:,0], size=(64, 64), Verbose=False, feature_vector=False)
	#hog_c2 = get_hog_features(roi[:,:,1], size=(64, 64), Verbose=False, feature_vector=False)
	#hog_c3 = get_hog_features(roi[:,:,2], size=(64, 64), Verbose=False, feature_vector=False)

	for x_cell in range(n_step_in_cells_x):
		for y_cell in range(n_step_in_cells_y):
			y_pos_in_cells = y_cell * n_cell_per_step
			x_pos_in_cells = x_cell * n_cell_per_step

			hog_feat_c1 = hog_c1[y_pos_in_cells: y_pos_in_cells + n_cell_per_window, x_pos_in_cells: x_pos_in_cells + n_cell_per_window].ravel()
			#hog_feat_c2 = hog_c2[y_pos_in_cells: y_pos_in_cells + n_cell_per_window, x_pos_in_cells: x_pos_in_cells + n_cell_per_window].ravel()
			#hog_feat_c3 = hog_c3[y_pos_in_cells: y_pos_in_cells + n_cell_per_window, x_pos_in_cells: x_pos_in_cells + n_cell_per_window].ravel()

			hog_feat = hog_feat_c1#np.concatenate([hog_feat_c1, hog_feat_c2, hog_feat_c3])

			xtl = x_pos_in_cells * pix_per_cell
			ytl = y_pos_in_cells * pix_per_cell

			# crop a window from the roi
			patch = cv2.resize(roi[ytl:ytl+window, xtl:xtl+window, :], (resize_h, resize_w))

			hist_feat = get_hist_feature(patch)

			spatial_feat = get_spatial_bin_feature(patch)

			desc = np.concatenate([hist_feat, spatial_feat, hog_feat])
			desc = desc.reshape(1, -1)

			scaled_desc = feature_scaler.transform(desc)

			y_pred = clf.predict(scaled_desc)

			if(y_pred == 1):
				print("y_pred = ", y_pred, " scale = ", scale)
				#resized = img_as_ubyte(resized)#resized.copy()
				x1 = np.int(xtl * scale)
				y1 = np.int(ytl * scale)
				winW = np.int(window * scale)
				winH = winW

				tl = (x1, y1 + y_start)
				br = (x1 + winW, y1 + winH + y_start)

				boxes.append((tl, br))

	return boxes



def pipeline2(img, clf, feature_scaler, verbose=False):
	scales = [1, 0.5]
	boxes = []
	result = img

	for scale in scales:
		boxes += detect_cars(img, scale, clf, feature_scaler)

	print("Boxes before NMS: ",len(boxes))

	#boxes_NMS = []
	#for box in boxes:
	#	boxes_NMS.append([box[0][0], box[0][1], box[1][0], box[1][1]])

	#boxes_clean = NMS(np.array(boxes_NMS), 0.3)
	#print("Boxes after NMS: ",len(boxes_clean))

	heatmap, heatmap_thresh = heat_map(img, boxes, thresh=0, verbose=False, nms=False)
	#cv2.imshow("heatmap", cv2.resize(heatmap, (400, 300)))
	#cv2.waitKey(0)
	#cv2.imshow("heatmap_thresh", cv2.resize(heatmap_thresh, (400, 300)))
	#v2.waitKey(0)
	# label connected components
	segmented_heat_map, num_objects = scipy.ndimage.measurements.label(heatmap_thresh)
	print("Num obj = ", num_objects)

	# prepare images for blend
	#img_hot_windows = img#draw_boxes(img, hot_windows, color=(0, 0, 255), thick=2)                 # show pos windows
	#img_heatmap = cv2.applyColorMap(normalize_image(heatmap), colormap=cv2.COLORMAP_HOT)         # draw heatmap
	#img_segmented_heat_map = cv2.applyColorMap(normalize_image(segmented_heat_map), colormap=cv2.COLORMAP_HOT)  # draw label
	#img_detection = draw_labeled_bounding_boxes(frame.copy(), labeled_frame, num_objects)        # draw detected bboxes
	result, final_boxes = draw_segments(result, segmented_heat_map, num_objects)
	#img_blend_out = prepare_output_blend(frame, img_hot_windows, img_heatmap, img_labeling, img_detection)

	if verbose:
	    #cv2.imshow('detection bboxes', cv2.resize(img, (480, 300)))
	    cv2.imshow('heatmap', cv2.resize(img_heatmap, (480, 300)))
	    cv2.imshow('img_segmented_heat_map', cv2.resize(img_segmented_heat_map, (480, 300)))
	    #cv2.imshow('result', cv2.resize(result, (480, 300)))
	    cv2.waitKey(0)
	#cv2.imshow('result', cv2.resize(result, (480, 300)))
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	return result





def sliding_window_core(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(int(image.shape[0] * 0.5), image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0], :])

def sliding_window(img, clf, feature_scaler, verbose=False):
	# loop over the image pyramid
	winW, winH = (64,64)
	stepSize = 32
	boxes = []
	result = img.copy()
	for (scale, resized) in enumerate(pyramid_gaussian(img, downscale=2)):
		if resized.shape[0] < 64 or resized.shape[1] < 64:#or scale >= 1:
			break
		# loop over the sliding window for each layer of the pyramid
		for (x, y, window) in sliding_window_core(resized, stepSize=stepSize // (2**scale), windowSize=(winW, winH)):
	    	# if the window does not meet our desired window size, ignore it
			if window.shape[0] != winH or window.shape[1] != winW:
				continue

	    	# THIS IS WHERE YOU WOULD PROCESS YOUR WINDOW, SUCH AS APPLYING A
	    	# MACHINE LEARNING CLASSIFIER TO CLASSIFY THE CONTENTS OF THE
	    	# WINDOW
			#print(window.shape)
			#gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
			#print(gray.shape)
			window = img_as_ubyte(window)
			#print(window)
			#cv2.imshow("win", (window))
			#cv2.waitKey(10)
			desc = get_descriptor(color_convert(window, 'YCrCb'))
			#feature_scaler = StandardScaler().fit(desc)
			desc_scaled = feature_scaler.transform(desc)
			y_pred = clf.predict(desc_scaled)
			
			# since we do not have a classifier, we'll just draw the window
			if(y_pred == 1):
				print("y_pred = ", y_pred, " scale = ", scale)
				#resized = img_as_ubyte(resized)#resized.copy()
				x2 = (2**scale) * x
				y2 = (2**scale) * y
				winW2 = (2**scale) * winW
				winH2 = (2**scale) * winH
				boxes.append([x2, y2, x2 + winW2, y2 + winH2])
				#cv2.rectangle(img, (x2, y2), (x2 + winW2, y2 + winH2), (0, 255, 0), 2)
				#cv2.imshow("Window", img)
				#cv2.waitKey(1)
			#cv2.waitKey(50)
		#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	#print("Before NMS: ", len(boxes))
	#boxes_clean = NMS(np.array(boxes), 0.35)
	#print("After NMS: ", len(boxes_clean))
	#for box in boxes_clean:
	#	cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
	#cv2.imshow("Window", img)
	#cv2.waitKey(0)

	heatmap, heatmap_thresh = heat_map(img, boxes, thresh=0, verbose=True)
	#cv2.imshow("heatmap", cv2.resize(heatmap, (400, 300)))
	#cv2.waitKey(0)
	#cv2.imshow("heatmap_thresh", cv2.resize(heatmap_thresh, (400, 300)))
	#v2.waitKey(0)
	# label connected components
	segmented_heat_map, num_objects = scipy.ndimage.measurements.label(heatmap_thresh)
	print("Num obj = ", num_objects)

	# prepare images for blend
	#img_hot_windows = img#draw_boxes(img, hot_windows, color=(0, 0, 255), thick=2)                 # show pos windows
	#img_heatmap = cv2.applyColorMap(normalize_image(heatmap), colormap=cv2.COLORMAP_HOT)         # draw heatmap
	#img_segmented_heat_map = cv2.applyColorMap(normalize_image(segmented_heat_map), colormap=cv2.COLORMAP_HOT)  # draw label
	#img_detection = draw_labeled_bounding_boxes(frame.copy(), labeled_frame, num_objects)        # draw detected bboxes
	result, final_boxes = draw_segments(result, segmented_heat_map, num_objects)
	#img_blend_out = prepare_output_blend(frame, img_hot_windows, img_heatmap, img_labeling, img_detection)

	if True:#verbose:
	    cv2.imshow('detection bboxes', cv2.resize(img, (480, 300)))
	    cv2.imshow('heatmap', cv2.resize(img_heatmap, (480, 300)))
	    cv2.imshow('img_segmented_heat_map', cv2.resize(img_segmented_heat_map, (480, 300)))
	    cv2.imshow('result', cv2.resize(result, (480, 300)))
	    cv2.waitKey(0)
	
	cv2.destroyAllWindows()


def get_args(name='default', video_file="None"):
	return video_file

filename = get_args(*sys.argv)

def pipeline(video_file):
	
	out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 25, (1280, 720))
	clf, feature_scaler = load_model()
	#sliding_window(cv2.imread('test_images/test4.jpg'), clf, feature_scaler)
	cap = cv2.VideoCapture(video_file)
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		# undistort the image
		result = pipeline2(frame, clf, feature_scaler)

		cv2.namedWindow("result", cv2.WINDOW_NORMAL)
		cv2.imshow("result", result)

		out.write(result)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

pipeline(filename)

def train():
	X_tr, X_val, y_train, y_val = load_data(bShuffle=True, cs='YCrCb')
	desc_tr = get_descriptor_for_train(X_tr)
	print(desc_tr.shape)

	desc_val = get_descriptor_for_train(X_val)
	print(desc_val.shape)

	clf, feature_scaler = train_svm(desc_tr, y_train)

	desc_val_scaled = feature_scaler.transform(desc_val)
	print('Validation accuracy: ', round(clf.score(desc_val_scaled, y_val), 4))

#train()