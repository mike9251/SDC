Vehicle Detection and Tracking Project:

1. ~~Load the data (X, y)~~
2. ~~Obtain feature vector~~
    ~~2.1 Historgrams for RGB channels~~
    ~~2.2 Downsampled image (unrolled)~~
    ~~2.3 HOG features~~
3. ~~Train SVM classifier with feature vectors~~
4. ~~Implement Sliding Window method~~
5. ~~Implement Heat Map calculation to eliminate false positive detections~~
6. ~~Try to use YOLO~~

Report:
1. Loading data
Loading data for training SVM classifier is implemented in `load_data.py` in `def load_data(bShuffle=False, cs='YCrCb')` function. It loads images both classes (vehicles, non-vehicles) and converts color space to `cs`. Also data augmentation is performed - each image is horizontally flipped, so the result data set is doubled. Finally the data set gets shuffled and splitted into train/val sets (90%/10%).
2. Descriptor  

    2.1 Histogtram (YCbCr)  
    Calculate histograms for each channel of the crop image (Window) and concatenate them into one vector. Use 256 bins,
    smaller number of bins affects SVM's classification.  
    
    2.2 Spatial bin  
    Resize the crop (Window) to `32x32` pixels and unravel it into a vector.  
    
    2.3 HOG. What it is, how to get, about parameters.  
    First of all gradient of the image is calculated. After calculating the gradient magnitude and orientation, we divide our
    image into cells and blocks.
    A “cell” is a rectangular region of pixels that belong to this cell. For example, if we had a 128 x 128 image and defined 
    our `pixels_per_cell` as 4x4, we would thus have 32 x 32 = 1024 cells.
     
    Then calculated histograms are normalized. For this step 'cells' are grouped into 'blocks'.	For each of the cells in the 
    current block we concatenate their corresponding gradient histograms, followed by either L1 or L2 normalizing the entire 
    concatenated feature vector. Normalization of HOGs increases performance of the descriptor.
    
    Parameters of the HOG descriptor extractor:  
    `orientations=9` - define the number of bins in the gradient histogram;  
    `pixels_per_cell=(8, 8)` - form cells with 8x8 pixels. HOG is calculated for each cell;  
    `cells_per_block=(2, 2)` - form blocks of cells, normalize each cell's HOG wrt the entire block;  
    `blocktransform_sqrt=True` - perform normalization;  
    `block_norm="L1"` -  perform L1 block normalization;  
    `feature_vector=True` - return descriptor in vector form (False - roi shape).  

    Instead of calculating HOG features for each Window in the image do it once for the entire image and then extract 
    features that correspond to the current Sliding Window. This approach improves performance.  

3. SVM, how it was trained, feature_scaler, save model
4. Sliding window - width, stride, scales
5. Heat map - what it is, how to calculate
6. YOLOv3
    6.1 Overview - different scales, number of anchers, how predictions are translated to pixels
    6.2 BBoxes, IoU, NMS
