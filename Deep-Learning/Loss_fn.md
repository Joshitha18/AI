
# Loss Functions

The ultimate goal of all algorithms of machine learning is to decrease loss. Loss has to be calculated before we try strategy to decrease it using different optimizers.

Loss function is sometimes also referred as Cost function.

Loss Function is an error in 1 data point while Cost Error Function is sum of all errors in a batch of dataset.


## Cost functions in Regression Problems

### Mean Squared Error

MSE measures the average of the sum of squares of the errors. It averages squared difference between the estimated values and the actual value. It is a kind of risk function where it calculates the deviation from the actual value with the predicted value which is squared and averaged with the number of instances a model has.

The MSE values closer to zero are better as this refers that model has less error.

#### Advantages

1. In the form of quadratic equation, when we plot a quadratic equation, we get a gradient descent with only one global minima.

2. There is no local minima.

3. It penalizes the model for making larger errors by squaring them. Example y-y^ is big then it will become bigger if it is squared.

#### Disadvantages

1. Outliers are not handled properly. As outlier error will be quite large, it is penalized squaring it.

### Mean Absolute Error

It can be called as arithmetic average of absolute errors, i.e. absolute difference between actual and predicted paired data points.

#### Advantages

1. Outliers are handled better than MSE as it is not penalizing the model by squaring error value.

#### Disadvantages

1. It is computationally expensive as it uses modulus operator function.

2. There may be a local minima.


## Classification Problems Loss functions

## Cross Entropy Loss

### Binary Cross Entropy = Sigmoid crossentropy

If you are training a binary classifier, then you may be using binary cross-entropy as your loss function.

Entropy as we know means impurity. The measure of impurity in a class is called entropy. SO loss here is defined as the number of the data which are misclassified.

We know that in binary classification problem Sigmoid function is used to calculate the output. Sigmoid which is used in logistic regression model for classification.

Unlike Softmax loss it is independent for each vector component (class), meaning that the loss computed for every CNN output vector component is not affected by other component values. That’s why it is **used for multi-label classification**, were the insight of an element belonging to a certain class should not influence the decision for another class. It’s called Binary Cross-Entropy Loss because it sets up a binary classification problem between C′=2 classes for every class in C, as explained above. 


### Categorical Cross-Entropy loss = softmax cross entropy loss

It is a Softmax activation plus a Cross-Entropy loss.
For Multiclass problems mostly Softmax function is used to classify the dataset. In turn, this means that the target variable must be one hot encoded.
This is to ensure that each example has an expected probability of 1.0 for the actual class value and an expected probability of 0.0 for all other class values.

### Sparse Multiclass Cross-Entropy Loss

A possible cause of frustration when using cross-entropy with classification problems with a large number of labels is the one hot encoding process.

For example, predicting words in a vocabulary may have tens or hundreds of thousands of categories, one for each label. This can mean that the target element of each training example may require a one hot encoded vector with tens or hundreds of thousands of zero values, requiring significant memory.

Sparse cross-entropy addresses this by performing the same cross-entropy calculation of error, without requiring that the target variable be one hot encoded prior to training.

[Ref](https://towardsdatascience.com/loss-functions-when-to-use-which-one-718ebad36e0)
