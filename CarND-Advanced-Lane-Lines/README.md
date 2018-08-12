## **Advanced Lane Lines Finding**

Overview
---

The goals of this project is to write a software pipeline to identify the lane boundaries in a video using more advanced algorithms (comparing to the simple one <a href=https://mike9251.github.io/2018/05/26/lane-line-detector/>Udacity Self-Driving Car Project #1</a>).<br>

Final video:
<iframe width="560" height="315" src="https://www.youtube.com/embed/Qm8VF2G83gU?ecver=1" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>


**Usage:** *python preprocess.py project_video.mp4*


Steps are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

**Camera calibration and distortion correction**

Pinhole cameras produce distorted images. With camera calibration we can determine the relation between the camera’s natural units (pixels) and the real world units (for example millimeters) and undistort the images. (More details in 
<a href="https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html">OpenCV documentation</a>).<br>
First of all we detect chessboard pattern in the calibrated images with `cv2.findChessboardCorners`.  Then using obtained 3d points in real world space and 2d points in image plane calculate the camera matrix and distortion matrix with `cv2.calibrateCamera` function. Finally, to undistort the image we use distortion matrix with `cv2.undistort` function.
<div class="imgcap">
 <img src="output_images/camera_calib_input.png" width="480" alt="Combined Image" /> 
 <div class="thecap">Distorted chessboard image</div>
</div>
<div class="imgcap">
 <img src="output_images/camera_calib_input1.png" width="480" alt="Combined Image" />
 <div class="thecap">Chessboard pattern</div>
</div>
<div class="imgcap">
 <img src="output_images/camera_calib_output.png" width="480" alt="Combined Image" />
 <div class="thecap">Undistorted chessboard image</div>
</div>
<div class="imgcap">
 <img src="output_images/input.png" width="480" alt="Combined Image" />
 <div class="thecap">Original input image</div>
</div>
<div class="imgcap">
 <img src="output_images/input_undist.png" width="480" alt="Combined Image" />
 <div class="thecap">Undistorted input image</div>
</div>
 
 **Creating a binary image**
 
 To create a binary representation of an image we will use several methods: extract yellow and white color masks, perform Sobel operator on the image and threshold the result. Then combined the results and perform morphological closing operation to highlight the lane lines. 
 <div class="imgcap">
 <img src="output_images/yw_img.png" width="480" alt="Combined Image" />
 <div class="thecap">Yellow/White color extract image</div>
 </div>
 <div class="imgcap">
 <img src="output_images/yw_img_thresh.png" width="480" alt="Combined Image" />
 <div class="thecap">Yellow/White color extract image binarized</div>
 </div>
 <div class="imgcap">
 <img src="output_images/sobel_mag.png" width="480" alt="Combined Image" />
 <div class="thecap">Sobel magnitude</div>
 </div>
 <div class="imgcap">
 <img src="output_images/sobel_mag_thresh.png" width="480" alt="Combined Image" />
 <div class="thecap">Binarized Sobel magnitude</div>
 </div>
 <div class="imgcap">
 <img src="output_images/bin_comb.png" width="480" alt="Combined Image" />
 <div class="thecap">Binarized image</div>
 </div>
 
 <div class="imgcap">
 <img src="output_images/preprocessed_roi.png" width="480" alt="Combined Image" />
 <div class="thecap">Preprocessed ROI</div>
 </div>
 
 **Applying perspective transform**
 <div class="imgcap">
 <img src="output_images/transform_input.png" width="480" alt="Combined Image" />
 <div class="thecap">Input image</div>
 </div>
 <div class="imgcap">
 <img src="output_images/transform_output.png" width="480" alt="Combined Image" />
 <div class="thecap">"birds-eye view" image</div>
 </div>
 
 **Detect the lane boundaries**
 <div class="imgcap">
 <img src="output_images/transformed_roi.png" width="480" alt="Combined Image" />
 <div class="thecap">"birds-eye view" image</div>
 </div>
 <div class="imgcap">
 <img src="output_images/boxes.png" width="480" alt="Combined Image" />
 <div class="thecap">Detecting the lane lines</div>
 </div>
 <div class="imgcap">
 <img src="output_images/lane_warp.png" width="480" alt="Combined Image" />
 <div class="thecap">Detected lane</div>
 </div>
 
 **Determine the curvature of the lane and vehicle position with respect to center.**
The curvature calculation method is described <a href="http://www.intmath.com/applications-differentiation/8-radius-curvature.php">here</a>. Assume that the camera is located at the center on the car. Then the car's deviation from the center of the lane can be calculated `|image_center - lane_center|`. To convert pixels to meters I used an assumption that the lane is about 30 meters long and 3.7 meters wide (provided by Udacity).

 **Warp the detected lane boundaries back onto the original image. Final result.**
 <div class="imgcap">
 <img src="output_images/lane.png" width="480" alt="Combined Image" />
 <div class="thecap">Unwarped Detected lane</div>
 </div>
 <div class="imgcap">
 <img src="output_images/result.png" width="480" alt="Combined Image" />
 <div class="thecap">Result image</div>
 </div>

