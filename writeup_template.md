# **Traffic Sign Recognition** 


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

[image1]: ./writeup-ref/dataset_visualize.png "Visualization"
[image2]: ./writeup-ref/TrafficSign_Color.png "Color"
[image3]: ./writeup-ref/augmented_image.png "Augmented Image"
[image4]: ./german-traffic-sign-test/german-road-signs-animals.png "Traffic Sign 1"
[image5]: ./german-traffic-sign-test/german-road-signs-traffic.png "Traffic Sign 2"
[image6]: ./german-traffic-sign-test/stop.jpg "Traffic Sign 3"
[image7]: ./german-traffic-sign-test/germany-speed-limit-sign.png "Traffic Sign 4"
[image8]: ./german-traffic-sign-test/no-overtaking-sign.png "Traffic Sign 5"
[image9]: ./writeup-ref/TrafficSign_grayscaled.png "GrayScaled"
[image10]: ./writeup-ref/AugmentedDataSet.png "Augmented Data set"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/singh-arulraj/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because color does not add any additional information.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]
![alt text][image9]


As a last step, I normalized the image data because we want to have our data to have zero mean. It makes optimizer more efficient.

I decided to generate additional data because the current data set is not enough to train the images.

To add more data to the the data set, I used the following techniques 
1. Gamma Cleaning
2. Rotate Slight left by 10 degrees
3. Rotate Slight Right by 10 degrees
4. Flip the image to the vertically (Only Specific Images)
5. Flip Image horizontally (Only Specific Images)
6. Flip Image both horizontally and vertically (Only Specific Images)
7. Adding noise to the image

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 
![alt text][image10]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6   				|
| Flattening            | Output size 400                               |  
| Fully connected		| Output size 120								|
| RELU					|												|
| Dropout				|												|
| Fully connected		| Output size 84								|
| RELU					|												|
| Dropout				|												|
| Fully connected		| Output size 43								|
| Softmax				|            									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used 
Optimiser        : AdamOptimizer
Batch Size       : 128
Number of epochs : 100
Learning Rate    : 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Training Data Accuracy of 98.2 %
* Validation set accuracy of 95.5%
* Test set accuracy of 92.9%

* LeNet architecture was chosen to classify the traffic signs.
* With LeNet architecture  could achieve an accuracy of around 95.5 % with augmented data. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Wild Animal Crossing	| Wild animals crossing 						| 
| Traffic Signals		| Traffic Signals								|
| Stop					| Beware of ice/ snow							|
| 60 km/h	      		| 80 km/h   					 				|
| No passing			| Vehicles over 3.5 metric tons prohibited		|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in section "Predict the Sign Type for Each Image" of the Ipython notebook.

For the first image, the model is relatively sure that this is a Wild Animal Crossing (probability of 0.99), and the image does contain a wild animal crossing . The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Wild Animal Crossing							| 
| .002    				| Keep Right									|
| .0005					| Go straight or Left							|
| .0002      			| Keep Left  					 				|
| .0002 			    | Slippery Road      							|


For the second image , the model is relatively sure that this is a Traffic Signals (probability of 0.74), and the image does contain a Traffic Signals. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .74         			| Traffic Signals   							| 
| .14    				| General Caution								|
| .045					| Pedestrian        							|
| .014      			| Double Curve 					 				|
| .011  			    | Danegrous Curve to Left						|


For the third image, the model is unsure. It predicts wrongly as Beware of Ice/Snow(0.14). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .14         			| Beware of Ice/Snow							| 
| .087    				| End of speed limit (80km/h)					|
| .075					| Speed limit (50km/h)							|
| .066      			| Double Curve 					 				|
| .057 			        | Turn left ahead      							|


For the Fourth image, the model is relatively sure that this is a Speed limit (80km/h) (probability of 0.58), but the image is Speed limit (60km/h). The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .58         			| Speed limit (80km/h)							| 
| .087    				| Speed limit (50km/h)							|
| .065					| Speed limit (30km/h)							|
| .059      			| Speed limit (100km/h)			 				|
| .04   			    | Roundabout mandatory 							|


For the fifth image, the model is relatively sure that this is a Vehicles over 3.5 metric tons prohibited (probability of 1.00), but the image is No passing. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00        			| Vehicles over 3.5 metric tons prohibited		| 
| .00   				| Roundabout mandatory							|
| .000					| No Passing         							|
| .000      			| No passing for vehicles over 3.5 metric tons	|
| .000  			    | End of no passing    							|



### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


