import numpy as np
import cv2
from skimage import feature, exposure
from skimage.transform import pyramid_gaussian
import matplotlib.pyplot as plt


def get_hist_feature(image, size=(64, 64), verbous=False):
    c1_hist = cv2.calcHist([image],[0],None,[256],[0,256])
    c2_hist = cv2.calcHist([image],[1],None,[256],[0,256])
    c3_hist = cv2.calcHist([image],[2],None,[256],[0,256])
    
    if(verbous == True):
        hist = np.concatenate((c1_hist, c2_hist, c3_hist), axis=1)
        color = ('c1','c2','c3')
        for i, col in enumerate(color):
            plt.plot(hist[:,i],color = col)
            plt.xlim([0,256])
        plt.show()
    
    return np.concatenate((c1_hist, c2_hist, c3_hist), axis=0)[:, 0]

def get_spatial_bin_feature(image, size=(16, 16)):
    return cv2.resize(image, size).ravel()

def get_hog_features(image, size=(32, 32), Verbose=False, feature_vector=False):
    #print("HOG: ", image.shape)
    #cv2.imshow("hist", image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #if(len(image.shape) == 3):
    #    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #if(size[0] != image.shape[0] or size[1] != image.shape[1]):
    #    gray = cv2.resize(gray, size)
    
    # extract Histogram of Oriented Gradients from the image
    (H, hogImage) = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), transform_sqrt=True,
                                block_norm="L1", visualize=True, feature_vector=feature_vector)
    
    if (Verbose == True):
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        plt.imshow(hogImage, 'gray')
        #cv2.imshow("HOG Image", hogImage)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
    return H

def get_descriptor(images):
    
    descriptors = []

    if(len(images.shape) <= 3):
        img_desc = []
        img_desc.append(get_hist_feature(np.uint8(images), size=(64, 64), verbous=False))
        img_desc.append(get_spatial_bin_feature(images))
        img_desc.append(get_hog_features(np.uint8(images), size=(64, 64), Verbose=False, feature_vector=True))
        
        descriptors.append(np.concatenate(img_desc))
    else:
	    for img in images:
	        img_desc = []
	        img_desc.append(get_hist_feature(img))
	        img_desc.append(get_spatial_bin_feature(img))
	        img_desc.append(get_hog_features(img, size=(64, 64), Verbose=False, feature_vector=True))
	        
	        descriptors.append(np.concatenate(img_desc))
    
    return np.array(descriptors)

def get_descriptor_for_train(images):
    
    descriptors = []

    for (i, img) in enumerate(images):
        img_desc = []
        img_desc.append(get_hist_feature(img))
        img_desc.append(get_spatial_bin_feature(img))
        img_desc.append(get_hog_features(img[:,:,0], size=(64, 64), Verbose=False, feature_vector=True))
        img_desc.append(get_hog_features(img[:,:,1], size=(64, 64), Verbose=False, feature_vector=True))
        img_desc.append(get_hog_features(img[:,:,2], size=(64, 64), Verbose=False, feature_vector=True))

        #if(i == 0):
        #	print("Shapes:\nHist = ", img_desc[0].shape, "\nSpatial = ", img_desc[1].shape, "\nHog1 = ", img_desc[2].shape, "\nHog2 = ", img_desc[3].shape, "\nHog3 = ", img_desc[4].shape)
        
        descriptors.append(np.concatenate(img_desc))
    
    return np.array(descriptors)