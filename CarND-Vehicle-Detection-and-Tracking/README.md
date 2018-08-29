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
1. About loading data
2. Descriptor
    2.1 Histogtram (YCbCr)
    2.2 Spatial bin
    2.3 HOG. What it is, how to get, about parameters.
3.  SVM, how it was trained, feature_scaler, save model
4. Sliding window - width, stride, scales
5. Heat map - what it is, how to calculate
6. YOLOv3
    6.1 Overview - different scales, number of anchers, how predictions are translated to pixels
    6.2 BBoxes, IoU, NMS
