
# Bias and Variance

![Bias and variance using bulls-eye diagram](https://miro.medium.com/max/1264/1*xwtSpR_zg7j7zusa4IDHNQ.png)

## What is bias?

Bias is how far the predicted values are from the actual values. Bias, in simple terms, is known as the difference between the average prediction of our model and the actual(correct) value. If the average predicted values are far off from the actual values then the bias is said to be high.
A case of high bias would be an example of the model under-fitting the dataset(poorly performing on the training set).

### **How to deal with it?**

1. Using a larger network (Adding more layers, Increasing the number of hidden units)
2. Using different Neural net Architecture.

## What is variance?

Usually, we build a model in such a way that it can make accurate predictions on the training set. If training and test sets have enough in common we expect the model to also be accurate on the test set. High Variance is the scenario in which the model gives very high accuracy over or very low error rate on the training set but fails entirely on the test or validation set. It is a type of error that occurs due to a model’s sensitivity to small fluctuations in the training set.

### **How to deal with it?**

1. Regularization
2. Dropout
3. Increasing the size of the dataset
4. Using algorithms like Early Stopping


# Regularization

Regularization means explicitly restricting a model to avoid overfitting. Stronger regularization pushes coefficients more and more toward zero, though coefficients never become zero exactly. Regularizations are of two types L1 regularization and L2 regularization.

![regularization](https://miro.medium.com/max/2100/1*pv8LrA_E7Npr3FXz9UvBmg.png)

A regression model that uses L1 regularization technique is called **Lasso Regression** and model which uses L2 is called **Ridge Regression**.

The key difference between these two is the penalty term.

Ridge regression adds “squared magnitude” of coefficient as penalty term to the loss function. Here, if lambda is zero then you can imagine we get back OLS. However, if lambda is very large then it will add too much weight and it will lead to under-fitting. Having said that it’s important how lambda is chosen. This technique works very well to avoid over-fitting issue.

Lasso Regression (Least Absolute Shrinkage and Selection Operator) adds “absolute value of magnitude” of coefficient as penalty term to the loss function.

Again, if lambda is zero then we will get back OLS whereas very large value will make coefficients zero hence it will under-fit.
The key difference between these techniques is that Lasso shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features.


# Dropout 

Dropout is an approach to regularization in neural networks which helps to **reduce interdependent learning amongst the neurons**. The parameter for a dropout function is dropout rate which can be defined as the probability of training a given node in a layer, where 1.0 means no dropout, and 0.0 means no outputs from the layer.


References:
[bias and variance](https://ai.plainenglish.io/bias-and-variance-3290edd0850d)
[regularization](https://towardsdatascience.com/l1-and-l2-regularization-methods-ce25e7fc831c)