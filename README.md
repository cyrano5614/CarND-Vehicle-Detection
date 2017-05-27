# P5 - Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

## Overview

In this project, we will explore and develop a pipeline to detect vehicles in given video or image.  The goals / steps of this project are the following:

* Extract features on a labeled training data provided by Udacity using various techniques including HOG, color transform, hisotrams of colors, and etc.
* Train the Classifier
* Use sliding-window technique to search for vehicles in a given image or video that is identified by classifier.
* Use heat map to reduce false positives and outliers by taking average of heap map over frames.
* Using the averaged heat map, identify bounding box for vehicles detected.
* Integrate Advanced Lane finding project result into the final pipeline.
* Implement the entire pipeline on a given image or video 

[//]: # (Image References)
[image1]: ./output_images/hog.png
[image2]: ./output_images/sliding_window.png
[image3]: ./output_images/heatmap.png
[image4]: ./output_images/video_thumbnail.jpg

---

## Histogram of Oriented Gradients (HOG) and feature extraction

Different color spaces such as RGB, YUV, LUV, and YCrCb was explored along with adjustment to HOG channels and other parameters.  After implementing and testing in several test images and videos for all different settings, optimized configuration was selected.  Here is example of exploring color space and HOG.

![alt text][image1]

The parameters with highest accuracy from classifier didn't necesarilly do best in the video.  For the pipeline, color space of 'YCrCb', HOG orientation of 9, HOG pixels per cell of 8, HOG cells per block of 2, and all channels of the 'YCrCb' was used for final features.  Below is a code for extracting hog features as well as visualization.  Code for this step can be found in pipeline.ipynb in functions section.

```
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
```

Histogram features and Spatial features were added on to the final features also for the final feature length of 6156.

## Selecting and Training the classifier

Sample size of 8000 cars and non-cars were used to extract features and train the classifier.  For future use, the features were extracted and saved to not run the extraction repeatedly.  After extracting the features, StandardScaler function from sklearn was used to normalize the extracted features from different methods so everything can have equal weight.

To define the labels for classifier, all cars were assigned to 1 and non-cars were assigned to 0.  The feature set was shuffled and divided into training and test set of 80:20 using train_test_split function from sklearn.

The LinearSVC classifier from sklearn was chosen for the final classifier as it easily achieved the accuracy of 99%.  The code for training can be found at pipeline.ipynb 'Train the Classifier' section.

## Sliding Window Search

Experimentation with set window size for sliding window search showed that it either missed or mis-classified vehicles.  To make the pipeline more robust, scaled window search was used.

![alt text][image2]

As seen on the image above, different sizes of windows were used at different locations of the image to adapt to varying size of vehicles on the image.  Having multiple sizes of windows overlapping on the 'Hot' spots where vehicles usually show up made the search more robust.  After running the scaled window search in various images and videos, scaled window sizes and overlaps were determined.

To reduce the false positives detected by the classifier and to put a bounding box to the detected vehicle's location, heatmap was implemented.  

![alt text][image3]

The number of positive hits by sliding window search made the boxes overlap which was converted into heatmap where it could be thresholded to eliminate the false positive and clean up the positive.  Label function from scipy library was used to label heatmap cluster to vehicle.  The code can be found at the 'Visualize heat map' sectioin in pipeline.ipynb file.

## Video Implementation

![Project video Output][image4]
(https://www.youtube.com/watch?v=UstIcRWLE0Q)

For the video implementation, average of heatmap over 10 frames were used to smooth out the bounding boxes for the detected vehicles and reduce false positives on the video.  Lane finding from previous project was also implemented in the video.

## Discussion

By increasing the number of search windows while scaled to different sizes gave the pipeline more robust detection of vehicles but it also increased the process time significantly.  To improve the model, optimizing the pipeline to speed the process up would be a priority.  

Few things to try in the future would be implementing HOG feature extration in the entire image and not the individual subsampling of slide windows.  This would speed up the process significantly.  We can also try to use neural network to identify vehicles which could be much faster.