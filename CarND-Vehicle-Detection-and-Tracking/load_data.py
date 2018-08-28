import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def color_convert(image, cs):
	if cs == 'HSV':
	    img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	elif cs == 'LUV':
	    img = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
	elif cs == 'HLS':
	    img = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
	elif cs == 'YUV':
	    img = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
	elif cs == 'YCrCb':
	    img = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

	return img


def load_data(bShuffle=False, cs='YCrCb'):
    X = []
    y = []
    
    vehicle_image_names = glob.glob('data/vehicles/*/*.png')
    #n_tr = [x for x in range(0, len(vehicle_image_names), len(vehicle_image_names) // 10)]
    print("Loading train dataset: \n")
    for (i, name) in enumerate(vehicle_image_names):
        X.append(color_convert(cv2.imread(name), cs))
        y.append(1)
        horizontal_img = cv2.flip(X[-1], 1 )
        X.append(horizontal_img)
        y.append(1)
        #if(i == n_tr.front):
        #	num = 10
        #	print(num, "% ")
        #	num = 2 * num
    
    non_vehicle_image_names = glob.glob('data/non-vehicles/*/*.png')
    print("\nLoading val dataset: \n")
    n_val = [x for x in range(0, len(vehicle_image_names), len(vehicle_image_names) // 10)]
    for name in non_vehicle_image_names:
        X.append(color_convert(cv2.imread(name), cs))
        y.append(0)
        horizontal_img = cv2.flip(X[-1], 1 )
        X.append(horizontal_img)
        y.append(0)

    if (bShuffle == True):
        X, y = shuffle(X, y)
        
    X_tr, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
        
    return np.array(X_tr), np.array(X_val), np.array(y_train), np.array(y_val)