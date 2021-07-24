[Logistic Regression Blog](https://www.analyticsvidhya.com/blog/2020/11/binary-cross-entropy-aka-log-loss-the-cost-function-used-in-logistic-regression/)

If z represents the output of the linear layer of a model trained with logistic regression, then sigmoid(z) will yield a value (a probability) between 0 and 1. In mathematical terms:

> y' = 1​/(1+ e^-z)
> y'is the output of the logistic regression model for a particular example.
> z = b + w1x1 + w2x2 + .. + wnxn
> The w values are the model's learned weights, and b is the bias.
> The x values are the feature values for a particular example.

Note that z is also referred to as the log-odds because the inverse of the sigmoid states that z can be defined as the log of the probability of the 1 label (e.g., "dog barks") divided by the probability of the 0 label (e.g., "dog doesn't bark"):

> z = log(y/(1-y))


## Why not MSE as Cost Function?

In Logistic Regression Ŷi is a nonlinear function(Ŷ = 1​/1+ e^-z), if we put this in the MSE equation it will give a non-convex function as shown:

![mse](https://editor.analyticsvidhya.com/uploads/13012download.jpg)

The cost function used in Logistic Regression is Log Loss.

Log Loss is the most important classification metric based on probabilities. It’s hard to interpret raw log-loss values, but log-loss is still a good metric for comparing models. For any given problem, a lower log loss value means better predictions.

Mathematical interpretation:

Log Loss is the negative average of the log of corrected predicted probabilities for each instance.

log loss:

![logloss](https://editor.analyticsvidhya.com/uploads/34447Capture.PNG)

There are three steps to find Log Loss:

- To find corrected probabilities.
- Take a log of corrected probabilities.
- Take the negative average of the values we get in the 2nd step.

If we summarize all the above steps, we can use the formula:-
![logloss](https://editor.analyticsvidhya.com/uploads/90149Capture0.PNG)

Here Yi represents the actual class and log(p(yi)is the probability of that class.

- p(yi) is the probability of 1.
- 1-p(yi) is the probability of 0.

Now Let’s see how the above formula is working in two cases:

- When the actual class is 1: second term in the formula would be 0 and we will left with first term i.e. yi.log(p(yi)) and (1-1).log(1-p(yi) this will be 0.

- When the actual class is 0: First-term would be 0 and will be left with the second term i.e (1-yi).log(1-p(yi)) and 0.log(p(yi)) will be 0.

We got back to the original formula for binary cross-entropy/log loss

![img](https://editor.analyticsvidhya.com/uploads/661483.png)


- The Red line represents 1 class. As we can see, when the predicted probability (x-axis) is close to 1, the loss is less and when the predicted probability is close to 0, loss approaches infinity.

- The Black line represents 0 class. As we can see, when the predicted probability (x-axis) is close to 0, the loss is less and when the predicted probability is close to 1, loss approaches infinity.
