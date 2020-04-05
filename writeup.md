# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

#### 2. Include an exploratory visualization of the dataset.



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I am feeding color images into the network, I tried grayscaleing but it did not improve the outcome. It also seemed counterintuitive that the color does not contain information.

As a last step, I normalized the image data to get a mean value of 0 and a range of -1 to +1.

The image training set is augmented by changing the brightness of the images in the training set so that every class has the same number of training images.
Other augmenting ideas e.g. flipping/mirroring/rotating did not improve the output what made sense to me because this can change the meaning/class of a traffic sign and thus does not improve classification. 

Here is an example of an original image and two augmented images:

<img src="WriteUp/image_orig_610.jpg" alt="image original"/>
<img src="WriteUp/image_brighter_610.jpg" alt="image brighter"/>
<img src="WriteUp/image_darker_610.jpg" alt="image darker"/>



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU    				|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten		      	| outputs 1x400 								|
| Fully connected		| input 400, output 120        					|
| Fully connected		| input 120, output 84        					|
| Fully connected		| input 84, output 43        					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used LeNet and most of the parameters from the previous exercise, i.e. 
- AdamOptimizer with a training rate of 0.0011
- EPOCHS = 50
- BATCH_SIZE = 256


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
INFO:tensorflow:Restoring parameters from ./lenet
Training Accuracy = 0.963


* validation set accuracy of ? 
INFO:tensorflow:Restoring parameters from ./lenet
Validation Accuracy = 0.873


* test set accuracy of ?
INFO:tensorflow:Restoring parameters from ./lenet
Test Accuracy = 0.882


I chose the LeNet architecture because it was introduced in the lecture and has proven to be sucessfull. The trained model can correctly identify most of the presented images.
 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<img src="Do-Not-Enter.jpg" alt="Do Not Enter"/>
<img src="No_speed_limit_sign.jpg" alt="No Speed Limit"/>
<img src="german_4.jpg" alt="60 km per h"/>
<img src="mifuUb0.jpg" alt="Roadwork"/>
<img src="traffic_light.jpg" alt="Traffic Light"/>


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Do Not Enter      	| Do Not Enter   								| 
| No Speed Limit     	| No Speed Limit 								|
| 60 km/h				| 60 km/h										|
| Roadwork	      		| Roadwork					 					|
| Traffic Light			| Traffic Light      							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 43th cell of the Ipython notebook. I think there is something wrong because the outcome looks too good the net is always 100% sure that the image was classified correctly.

For the first image, the model is relatively sure that this is a stop sign (probability of 1.0), and the image does contain a Do-Not-Enter sign. The top five soft max probabilities were

| Probability		    |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0  		    		| Do Not Enter   								| 
| 0.0	     			| No Speed Limit 								|
| 0.0					| 60 km/h										|
| 0.0	      			| Roadwork					 					|
| 0.0					| Traffic Light      							|


For the second image ... 
| Probability		    |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.0  		    		| Do Not Enter   								| 
| 1.0	     			| No Speed Limit 								|
| 0.0					| 60 km/h										|
| 0.0	      			| Roadwork					 					|
| 0.0					| Traffic Light      							|

For the third image ... 
| Probability		    |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.0  		    		| Do Not Enter   								| 
| 0.0	     			| No Speed Limit 								|
| 1.0					| 60 km/h										|
| 0.0	      			| Roadwork					 					|
| 0.0					| Traffic Light      							|

For the fourth image ... 
| Probability		    |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.0  		    		| Do Not Enter   								| 
| 0.0	     			| No Speed Limit 								|
| 0.0					| 60 km/h										|
| 1.0	      			| Roadwork					 					|
| 0.0					| Traffic Light      							|

For the fifth image ... 
| Probability		    |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.0  		    		| Do Not Enter   								| 
| 0.0	     			| No Speed Limit 								|
| 0.0					| 60 km/h										|
| 0.0	      			| Roadwork					 					|
| 1.0					| Traffic Light      							|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

In the visual output the horizontal bar of the Do-Not_Enter sign is highlighted. 

<img src="feature_map_0.jpg" alt="Feature map 0"/>
<img src="feature_map_1.jpg" alt="Feature map 1"/>
<img src="feature_map_2.jpg" alt="Feature map 2"/>
<img src="feature_map_3.jpg" alt="Feature map 3"/>
<img src="feature_map_4.jpg" alt="Feature map 4"/>
<img src="feature_map_5.jpg" alt="Feature map 5"/>
<img src="feature_map_6.jpg" alt="Feature map 6"/>
<img src="feature_map_7.jpg" alt="Feature map 7"/>
<img src="feature_map_8.jpg" alt="Feature map 8"/>
<img src="feature_map_9.jpg" alt="Feature map 9"/>
<img src="feature_map_10.jpg" alt="Feature map 10"/>
<img src="feature_map_11.jpg" alt="Feature map 11"/>
<img src="feature_map_12.jpg" alt="Feature map 12"/>
<img src="feature_map_13.jpg" alt="Feature map 13"/>
<img src="feature_map_14.jpg" alt="Feature map 14"/>
<img src="feature_map_15.jpg" alt="Feature map 15"/>
