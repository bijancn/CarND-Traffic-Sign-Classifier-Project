# **Traffic Sign Recognition**

## Writeup

[//]: # (Image References)

[countplot]: ./countplot.png
[distplot]: ./distplot.png
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: 30.jpg "Traffic Sign 1"
[image5]: 70.jpg "Traffic Sign 2"
[image6]: nopassing.jpg "Traffic Sign 3"
[image7]: stop.jpg "Traffic Sign 4"
[image8]: straightright.jpg "Traffic Sign 5"

## Rubric Points
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set
I used plain Python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

I used Seaborn to get the distribution of images across the
different traffic sign classes:

![countplot][countplot]

The basic count plot above shows that the samples are far from
equidistributed. Therefore, training might overfit towards those signs
that are more often represented. For a general classifier, we should
thus augment the data by adding rotated or shifted images of the rare
classes.

However, as we can see from the dist plot below, the distribution is
roughly the same across training, validation and test sets. Thus, the
overfitting might actually help for these specific sets as it is more
likely to classify a frequent class.

![distplot][distplot]

### Design and Test a Model Architecture

#### 1. Preprocessing

To roughly normalize the mean and variance of the images, I subtract the
mean and divide by the mean in the normalize step. I am not sure if this
is ideal for testing single images like we do in the last step but in
the training, test and validation sets I found that their mean is closer
to 80 than 128.

I thought about using grayscale but I refrained from it as it is
counterintuitive for me to disregard the color information for German
traffic signs where color is in general relevant.

Finally, I shuffled the data and the labels. This, of course, only
affects the training but not the validation and test.

#### 2. Model architecture

My final model is almost identical to LeNet. It consists of the
following layers:

| Layer                 |     Description                             |
|:---------------------:|:---------------------------------------------:|
| Input                 | 32x32x3 RGB image                           |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 28x28x6  |
| RELU activation       |                                             |
| Max pooling           | 2x2 stride, valid padding, outputs 14x14x6  |
| Convolution 5x5       | 1x1 stride, valid padding, outputs 10x10x32 |
| RELU activation       |                                             |
| Max pooling           | 2x2 stride, valid padding, outputs 5x5x32   |
| Flatten               | outputs 800                                 |
| Fully connected       | outputs 120                                 |
| RELU activation       |                                             |
| Fully connected       | outputs 43                                  |
| Softmax               | etc.                                        |

As an experiment, I increased the shape of the randomly selected values
form 16 to 32 in the second convolution layer, which gave me a slight
improvement of 1-2 %.

#### 3. Training

The logits from the LeNet type architecture are used to compute the
Softmax-Cross-Entropy. The mean of this is minimized by the optimizer.
I used a learning rate of 0.001 with the `AdamOptimizer`. I set a
maximal number of epochs to 50 with a batch size of 56.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

As a validation accuracy of at least 93 % was required, I decided to
stop the optimization as soon as 95 % was reached.

My final model results were:
* validation set accuracy of 0.954
* test set accuracy of 0.934

As a starting point, I simply copied the LeNet solution as classifying
numbers should not be too different from classifying traffic signs and
it was already introduced in the lectures.

This gave me already a decent performance of ~85%. The first
hyperparameter I experimented with were epochs and batch sizes. Higher
batch sizes (512) did not improve the learning so I settled with 56.
I went with the epochs to up to 50 but this gave little improvements
over about 25. Finally, I adapted one of the dimension of the
convolution as noted above to reach the quoted accuracies.

The different type of convolutions seem to help to identify features at
different scales.

### Test a Model on New Images

#### 1. Five German traffic signs found on the web

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .60         			| Stop sign   									|
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ...

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


