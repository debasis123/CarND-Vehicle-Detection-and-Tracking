# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, our goal is to write a software pipeline to detect vehicles in a video.


The Project
---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* In addition, apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


[//]: # (Image References)
[image1]: ./output_images/sample.png "Sample car notcar images"
[image2]: ./output_images/bin_spatial.png "Bin spatial"
[image3]: ./output_images/color_spaces.png "Color spaces"
[image4]: ./output_images/col_hist.png "Color histogram"
[image5]: ./output_images/hog.png "Hog example"
[image6]: ./output_images/all_features.png "All features of an image"
[image7]: ./output_images/multiscale.png "multiscale search"
[image8]: ./output_images/heatmap.png "Heatmap, threshold, label"
[image9]: ./output_images/detection.png "Final boxes around cars"
[image10]: ./output_images/test_1.png "Final bounding boxes around cars"
[image11]: ./output_images/test_2.png "Final bounding boxes around cars"
[video1]: ./project_video_output.mp4 "My output video"


---

### Dataset exploration

The dataset provided by Udacity has the following statistics:
* no. of car images:  8792
* no. of notcar images:  8968
* no. of total images:  17760
* size of a car img:  (64, 64, 3)
* size of a noncar img:  (64, 64, 3)

Some random images of the cars and notcars are as follows:

![alt text][image1]


### Feature extraction

To get the training data from the dataset, I performed 3 types of feature extraction, viz.,
* Binned color features
* Color histogram features
* Histogram of Oriented Gradients (HOG) features

The implementation of this section can be found in the jupyter notebook under the heading `Extract various image features`.

Following is a visual representation of the difference between a car and a notcar image when applied binned color feature extraction. I used `(32, 32)` spatial sampling size.

![alt text][image2]

There is a clear distinction for this pair of images and so was included in my pipeline for the generation of training data.

For color histogram feature extraction, I looked into the various color spaces for an image to identify which one separates the car from the road better. Here is an illustration.

![alt text][image3]

After doing some more experimentation with the car and notcar images for various color spaces, I chose HSV color space for my pipeline. It is also evident from the above picture that HSV does a good job in this. I chose `32` histogram bins. Following is visual representatino of car and notcar features for color_histogram.

![alt text][image4]



### Histogram of Oriented Gradients (HOG)

<!-- #### 1. Explain how (and identify where in your code) you extracted HOG features from the training images. -->

`get_hog_features()` function under the `Hog features` subsection in the jupyter notebook implements HOG features. Since I was interested to extract features from all the channels, I implemented a utility function `hog_features_for_image()` for this as well.

I then explored different color spaces (HSV, YCrCb, HLS) along with different `skimage.hog()` parameters (`orientations` (8,9,11), `pixels_per_cell` (16 each, 32 each), and `cells_per_block` (1 each, 2 each)). Finally, I settled with the following values for hog parameters and HSV color space as before.
```
orientations = 9
pixels_per_cell = (8, 8)
cells_per_block = (2, 2)
```
I also tried random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like. Here is an example using the `HSV` color space. There is a clear pattern for the car image for each of the channels.

![alt text][image5]


Finally, `single_img_features()` function contains the feature extraction pipeline for a single image. Here is an example after applying the pipeline on a car and notcar image.

![alt text][image6]



### Training Classifier

Now I am ready to train a classifier with the training data (image features for each car and notcar image). `classify()` function in the `Classifier and normalization of images` section in the jupyter notebook implements the linear support vector machine. I did not experiment with the value of C and used the default one.
```python
LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
     verbose=0)
```
The classifer was trained on a feature vector length of 8460 and validation accuracy was 0.9837. I used the standard 80%-20% rule for the training/validation data.



### Sliding Window Search

The `Sliding window search` section in the jupyter notebook implements the efficient approach mentioned in the lesson to perform HOG feature extraction along with sliding window. I performed other feature extractions as well, in the same order as we did for training (by, `np.hstack((spatial_features, hist_features, hog_features))`).

For the purpose of identifying cars at various places of the image, I implemented `multiple-scaled windows` search. I took into consideration that the cars are bigger in size towards the bottom of the image and smaller towards the top. I did this through different values of scaling factor to be applied for window size for sliding window protocol. Finally all these windows merged into a single list.

Here are the paramters for various window sizes, chosen after numerous amount of iterations.
```
|  y_start  |  y_stop   |  scale |
|-----------|-----------|--------|
|  380      |  480      |  1.0   |
|  420      |  500      |  1.0   |
|  400      |  500      |  1.5   |
|  430      |  560      |  2.0   |
|  460      |  650      |  2.5   |
|  400      |  656      |  3.5   |
```
and here is how the window looks on the cars for test image test5.jpg when the above multiple-scaled search is applied.

![alt text][image7]



### Removing false positives

While running the sliding window at this stage on the test images, I noticed that there are many windows on the image where there is no car (falso positive). To get rid of them, I used heatmap. I used a threshold value `2` to get rid of intermittent windows (noise) in the frames. `Removing false positives` section in the jupyter notebook implements this. Following image shows the heatmap before and after applying this threshold on the above image. It also shows the label on the heatmaps (basically, the number of labels is the number of cars) while I apply scipy API `scipy.ndimage.measurements.label()` on the heatmap.

![alt text][image8]

Here is the resulting bounding boxes drawn onto the above test image after the labeling.

![alt text][image9]



### Pipeline

The pipeline for each image consisted of the following steps:
* applying the multiple scale search windows protocol
* applying heatmap on the image with the help of the windows from search
* threshold the heatmap
* apply scipy label() on the resultant heatmap
* draw labeled boxes around the cars in the image

Here are six test images and their corresponding heatmaps and labels on the car.

![alt text][image10]
![alt text][image11]
---


### Video Implementation

Here is the [link to my video result](./project_video_output.mp4)

---


### Discussion

* I still could find some false positives on the frames. Possibly it requires even more exhaustive search for the right parameters and hyperparameters.
* Another approach could be to maintain a list of windows for consecutive frames and use it for the next frame. That may reduce the intermittent noise in the frames.
* False positive was relatively high in the video in the shadowy road.
* Deep learning!

