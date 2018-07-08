## Writeup
### Kinji Sato  8th/July/2018

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/hog_examples.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). 
Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

It was difficult to evaluate the performance of HOG with different parameter valuse form the HOG images. So, I decided to evaluate those from the accuracy of LinerSVC to decide vehicle or not vehicle. Followig are the parameter set and the accuracy from LinerSVC.

| No. | Color space | Orientations | Pixels per cells | Cells per block | HOG channel | Accuracy from LinearSVC 
|:---:|:-----------:|:------------:|:----------------:|:---------------:|:-----------:|:----------------------:|
| 1		| RGB         | 9            | 8                | 2               | ALL         | 0.9257
| 2		| HSV         | 9            | 8                | 2               | 0           | 0.9003
| 3		| HSV         | 9            | 8                | 2               | 1           | 0.9046
| 4		| HSV         | 9            | 8                | 2               | 2           | 0.9175
| 5		| HSV         | 9            | 8                | 2               | ALL         | 0.9614
| 6		| LUV         | 9            | 8                | 2               | 0           | 0.9203
| 7		| LUV         | 9            | 8                | 2               | 1           | Memory error
| 8		| LUV         | 9            | 8                | 2               | 2           | Memory error
| 9		| LUV         | 9            | 8                | 2               | ALL         | Memory error
| 10	| HLS         | 9            | 8                | 2               | 0           | 0.8953
| 11	| HLS         | 9            | 8                | 2               | 1           | 0.9195
| 12	| HLS         | 9            | 8                | 2               | 2           | 0.8981
| 13	| HLS         | 9            | 8                | 2               | ALL         | 0.9555
| 14	| YUV         | 9            | 8                | 2               | 0           | 0.9257
| 15	| YUV         | 9            | 8                | 2               | 1           | 0.8986
| 16	| YUV         | 9            | 8                | 2               | 2           | 0.9336
| 17	| YUV         | 9            | 8                | 2               | ALL         | 0.9642
| 18	| YCrCb       | 9            | 8                | 2               | 0           | 0.9158
| 19	| YCrCb       | 9            | 8                | 2               | 1           | 0.9243
| 20	| YCrCb       | 9            | 8                | 2               | 2           | 0.8913
| 21	| YCrCb       | 9            | 8                | 2               | ALL         | 0.9665


From the above results, I can say HOG channel = `ALL` give better results than others. Unfortunately I don't know why memory erro occured when I ran with color space = `LUV`. `RGB` gave lower score than others.
Next, I evaluste different `orientations` value at the case of `HSV`, `HLS`, `YUV` and `YCrCb`

| No. | Color space | Orientations | Pixels per cells | Cells per block | HOG channel | Accuracy from LinearSVC 
|:---:|:-----------:|:------------:|:----------------:|:---------------:|:-----------:|:----------------------:|
| 1		| HSV         | 6            | 8                | 2               | ALL         | 0.9578
| 2		| HSV         | 12           | 8                | 2               | ALL         | 0.955
| 3		| HLS         | 6            | 8                | 2               | ALL         | 0.962
| 4		| HLS         | 12           | 8                | 2               | ALL         | 0.9513
| 5		| YUV         | 6            | 8                | 2               | ALL         | 0.9665
| 6		| YUV         | 12           | 8                | 2               | ALL         | 0.9606
| 7		| YCrCb       | 6            | 8                | 2               | ALL         | 0.9682
| 8		| YCrCb       | 12           | 8                | 2               | ALL         | 0.964


From above results, different orientation value gave small impact on the accuracy. So, I choosed `9` for `orientation` as same as the lecture video. At the case of `Color space` of `HSV` and `HLS`, those results were lower than others, so I choose `YUV` and `YCrCb` for next step.

Next, I evaluated different `Pixcles per cells` for `YUV` and `YCrCb`.

| No. | Color space | Orientations | Pixels per cells | Cells per block | HOG channel | Accuracy from LinearSVC 
|:---:|:-----------:|:------------:|:----------------:|:---------------:|:-----------:|:----------------------:|
| 1		| YUV         | 9            | 4                | 2               | ALL         | Memory error
| 2		| YUV         | 9            | 16               | 2               | ALL         | 0.9794
| 3		| YCrCb       | 9            | 4                | 2               | ALL         | Memory error
| 4		| YCrCb       | 9            | 16               | 2               | ALL         | 0.9797

When the `Pixcels per cells` are low value, those took much computation time and memory error occured. Higher value gave better results than lower value. I'm not sure how much effect on later project...
`YCrCb` gave better results constantly than other color space, so I choosed `YCrCb` for color space, and for other parameters, I choosed the defalut value those used in the lecuture video.


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

From above evaluation, I choosed following parameters for training of LinerSVC classifier.
And I also activated spatial_feat and hist_feat for this training.

```python
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size =(16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
```

From above parameter set, I got accuracy > 0.99 (99%). One of the reason I used `Pixcels per cellsn = 8` same as lecture video was from this 0.99 accuracy with spatial and coloer hist.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

