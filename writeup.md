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
[image3]: ./output_images/sw1.png
[image4]: ./output_images/sw2.png
[image5]: ./output_images/sw3.png
[image6]: ./output_images/sw4.png
[image7]: ./output_images/sw_all.png
[image8]: ./output_images/sw_fp.png
[image9]: ./output_images/labels2.png
[image10]: ./output_images/heatmap2.png
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

### Histogram of Oriented Gradients (HOG)

Here is a link to my [project code](https://github.com/kinjisato/CarND-Vehicle-Detection/blob/master/P01_HOG_005.ipynb).

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

The code for this extraction is in cell 12.

```python
img = img_car
features_car = []
hog_image_car = []
for channel in range(img.shape[2]):
    features, hog_image = get_hog_features(img[:,:,channel], orient, pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=True)
    features_car.append(features)
    hog_image_car.append(hog_image)

img = img_notcar
features_notcar = []
hog_image_notcar = []
for channel in range(img.shape[2]):
    features, hog_image = get_hog_features(img[:,:,channel], orient, pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=True)
    features_notcar.append(features)
    hog_image_notcar.append(hog_image)
```


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

The code for training is in cell 17 and 18 in my iPython notebook.


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

From above parameter set, I got accuracy > 0.99 (99%). One of the reason I used `Pixcels per cellsn = 8` same as lecture video was from this 0.99 accuracy with spatial and coloer hist. And I thought this accuray should be ok for next.

### Sliding Window Search

Here is a link to my [project code](https://github.com/kinjisato/CarND-Vehicle-Detection/blob/master/P02_SlidingWindow_003.ipynb).

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I did this step by step. First, I read the my LinerSVC classifer parameter from previous HOG extraction activity. 
I was very confused that even if I used the same SVC parameters I loaded from the same pickle file, the classifier result from the same sliding window were different at each time when I close and open my jupyter notebook.

From many many traials of window size (overlap was 75% same as lecture video) and number of false posirive, I decided following window size, start and stop positions.

| No. | ystart | ystop | scale, window size | Overlap
|:---:|:------:|:-----:|:------------------:|:-----------------:|
| 1	  | 400    | 480   | x1.0, 64 x 64      | 75%
| 2	  | 400    | 496   | x1.5, 96 x 96      | 75%
| 3	  | 400    | 528   | x2.0, 128 x 128    | 75%
| 4	  | 400    | 560   | x2.5, 160 x 160    | 75%
| 5	  | 464    | 656   | x3.0, 196 x 196    | 75%


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Following images were the results of my sliding windows and classifier.
(at this test image, No.5 window did not detect cars)

The code is in cell 19 to 26.

![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]

All window combined:
![alt text][image7]

Other test image
![alt text][image8]

To delete false positive, I increased the threshold of heatmap to '3'. (code cell 27)
And then I got following label and heatmap for this image.

![alt text][image9]
![alt text][image10]

---

### Video Implementation

Here is a link to my [project code](https://github.com/kinjisato/CarND-Vehicle-Detection/blob/master/P02_SlidingWindow_003.ipynb).
(The iPython notebook is the same as above "sliding window" activity)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

To deliete false positives from each video flame, heat map threshold was not enoguh. So, I made a filter (python class).
The code is in cell 31, and the code is,

```python
# Fileter
class box_filter():
    # initialize
    def __init__(self):
        self.prev_boxes = [] 
    
    
    def add_boxes(self, new_boxes):
        # appeend new boxes
        self.prev_boxes.append(new_boxes)
        # if previous boxes have more than n flames, delete old boxes
        if len(self.prev_boxes) > 10:
            self.prev_boxes = self.prev_boxes[len(self.prev_boxes)-10:]
```

This filter append and store the boxes of previous 10 flames of video. And the heat map theshold was,

```python
heat = apply_threshold(heat,1+len(SW_boxes_filter.prev_boxes)*4)
```

This threshold was tuned with the results of ouput video. To delite the most of false positives, I used this value. Maybe this filter was strong and even if the sliding window was detecting corretly, small number of heat were also delited. But most of the time in the video, this looked working well.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

### HOG and classifier
Maybe, there are many parameter combinations of HOG, spatial, hist...for feature extraction. And some combinations would give high accuracy for classifier. (This is from my evaluation results) If I have much time to check and evaluate the performance of parameter combination, that would be good. But, actualy, it is difficult to search all of the parameter combination. Is there any more effectie search for choosing paramter values?

### Sliding window
Usualy, I use Mac for my work. But, when I need to do some work that requires more computation performance, I use Windows PC. Is there any problem when I use pickle file in Mac, that was saved in Windows PC? Because as I explained above, when I run my sliding window, sometimes the result was much different. (My LinerSVC had 99% accuracy...)

I don't have so much experience of this sliding window search. It was very difficult to decide window size and start and stop positions. So, I used x1.0 to x 3.0 size of windows, and combined all for heat map. But maybe, these many boxes would increase the computation cost. And maybe, too much windows would make false positives. So, I'd like to reduce the number of sliding windows. And my strong filter would make the detection robust, but sometimes loose white car. How can I improve?


I'm very happy if I can have many advise to improve the all of the performance (including computation cost of PC).
