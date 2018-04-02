# **Traffic Sign Recognition**

[//]: # (Image References)

[countplot]: ./countplot.png
[distplot]: ./distplot.png
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: 30.jpg "Traffic Sign 1"
[image5]: 70.jpg "Traffic Sign 2"
[image6]: nopassing.jpg "Traffic Sign 3"
[image7]: stop.jpg "Traffic Sign 4"
[image8]: straightright.jpg "Traffic Sign 5"

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

#### 4. Solution finding

As a validation accuracy of at least 93 % was required, I decided to
stop the optimization as soon as 95 % was reached.

My final model results were, as it can be seen in the tensor flow session
with the epochs:
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

The signs should all be pretty easy to identify. Only the viewing angle
varies otherwise they are clearly visible. Potentially problematic is
the blue on blue of the straight-right, which offers little contrast.

#### 2. Model performance on the signs

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| 30 km/h     			| 30 km/h 										|
| 70 km/h					| 70 km/h											|
| No passing	      		| No passing					 				|
| Straight-Right			| Roundabout mandatory     							|

The model was able to correctly guess 4 of the 5 traffic signs, which
gives an accuracy of 80%. This is not as good as the performance on the
test set (93.4 %) but also not statistically significant due to the
small sample size.

#### 3. Model certainty

The top 5 softmax probabilities are computed with
`tf.nn.top_k(tf.nn.softmax(logits), k=5)`. However, as it can be seen
from the values of `logits` this is almost pointless as all of the
images have one `logit > 500`, leading to giant numbers when
exponentiated. Thus, all "probabilities" are extremely close to one.
However, I do not understand why the applying the softmax function
actually gives a probability as I could have used plenty of other
functions that normalizes to a number between 0 and 1. The softmax
function does not work with logits that large. The other question is
obviously why the logits are so large, which I cannot explain.

The candidates are
```
[14,  5,  1,  0, 13]
[ 9, 41, 35, 40, 16]
[ 4, 37,  1, 40, 18]
[ 1,  6,  2, 38,  5]
[40,  1, 36, 41, 13]
```
Thus the straight-right (36) is not only mistaken for a roundabout (40,
also white arrows on blue round sign) but also lands behind Speed limit
(30km/h) (1), which looks quite differently.

As mentioned in the preprocessing, for a more robust classifier we
should add more augmented data. Furthermore, the architecture is just
what I ended up with but can of course be improved by adding layers.
Especially, a drop-out might avoid the overfitting that is likely
indicated by the big logits.
