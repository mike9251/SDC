# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />

Overview
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project we will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

Writeup
---
The first task of the Self-Driving Car course is to create a simple lane line detector. The implemented script works with video only

*1. Pipeline*
 - load an image
 <img src="result/input_img.JPG" width="480" alt="Combined Image" />
 - preprocessing in HSL color space - IMPORTANT - really helps in the 'challenge' video case
 <div class="imgcap">
 <img src="result/input_img_hls.JPG" width="480" alt="Combined Image" />
 <div class="thecap">Input image in HSL color space</div>
 </div>
Extract white and yellow lines with corresponding masks:
<div class="imgcap">
 <img src="result/white_mask.JPG" width="480" alt="Combined Image" />
 <div class="thecap">Mask for white color</div>
 </div>
 <div class="imgcap">
 <img src="result/yellow_mask.JPG" width="480" alt="Combined Image" />
 <div class="thecap">Mask for yellow color</div>
 </div>
 Combine extracted images:
 <div class="imgcap">
 <img src="result/yellow_mask.JPG" width="480" alt="Combined Image" />
 <div class="thecap">Yellow + White mask</div>
 </div>
 Apply this mask to the input image to extract parts of it:
 <img src="result/extract_y_w.JPG" width="480" alt="Combined Image" />
 
 - convert the result of the previous stage into a gray scale
 <img src="result/gray.JPG" width="480" alt="Combined Image" />
 
 - perform filtering with the Gaussian filter
 <img src="result/blured.JPG" width="480" alt="Combined Image" />
 
 - find edges with the Canny edge detector
 <img src="result/edges.JPG" width="480" alt="Combined Image" />
 
 - define a region of interest (a region where we'll look for lane lines)
 <img src="result/roi.JPG" width="480" alt="Combined Image" />
 
 - find lines from the edge points with the Hough algorithm
 Hough algorithms returns an array of points: `x1 y1 x2 y2`. In my case I get 9 lines:
 ```
 [[[227 513 424 359]]

 [[213 512 374 391]]

 [[550 347 838 513]]

 [[529 338 821 513]]

 [[379 389 421 357]]

 [[292 461 407 371]]

 [[373 393 405 368]]

 [[585 368 747 465]]

 [[529 340 746 465]]]
 ```
 - separate found lines on left and right lines, calculate an average left/right lines
 To do so calculate slope of every line, if a slope is negative - it is a left line, if positive - right.
 Average left/right lines:
 ```left line:  ((186, 540), (466, 324)) right_line:  ((874, 540), (506, 324))```
 
 - draw lane lines
 <img src="result/lane_line.JPG" width="480" alt="Combined Image" />
 
 **Usage:** *python pipeline.py test_videos\solidYellowLeft.mp4*

*2. Identify any shortcomings*

Current implementation suffers to accurately identify curved lane lines.

*3. Suggest possible improvements*

-One possible way to improve accuracy of lane line detection is to apply extraction masks (white, yellow) to the input image - Done.
This method really helps, detector more accurately finds lines.

- To increase accuracy of curved line detection try to apply some different methods for outlier lane line rejection.

