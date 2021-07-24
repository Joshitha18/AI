
# What is an Activation Function?

Activation functions are mathematical equations that determine the output of a neural network.(*also called Transfer Function*) It defines how the weighted sum of the input is transformed into an output from a node or nodes in a layer of the network. 

All hidden layers typically use the same activation function. The output layer will typically use a different activation function from the hidden layers and is dependent upon the type of prediction required by the model.

Activation functions are also typically differentiable, meaning the first-order derivative can be calculated for a given input value. This is required given that neural networks are typically trained using the backpropagation of error algorithm that requires the derivative of prediction error in order to update the weights of the model.

An additional aspect of activation functions is that they must be computationally efficient because they are calculated across thousands or even millions of neurons for each data sample. Modern neural networks use backpropagation to train the model, which places an increased computational strain on the activation function, and its derivative function.


The basic process carried out by a neuron in a neural network is:
![image](https://miro.medium.com/max/2100/0*sykx-fOwxE0AeAMv.png)

The Activation Functions can be basically divided into 2 types-
- Linear Activation Function
- Non-linear Activation Functions

# Linear Activation Functions

> A = cx

It takes the inputs, multiplied by the weights for each neuron, and creates an output signal proportional to the input. 

## Two major problems:

1. Not possible to use backpropagation (gradient descent) to train the model — the derivative of the function is a constant, and has no relation to the input, X. So it’s not possible to go back and understand which weights in the input neurons can provide a better prediction.

2. All layers of the neural network collapse into one — with linear activation functions, no matter how many layers in the neural network, the last layer will be a linear function of the first layer (because a linear combination of linear functions is still a linear function). So a linear activation function turns the neural network into just one layer.

*A neural network with a linear activation function is simply a linear regression model. It doesn’t help with the complexity or various parameters of usual data that is fed to the neural networks.*

## Why derivative/differentiation is used ?

    When updating the curve, to know in which direction and how much to change or update the curve depending upon the slope.That is why we use differentiation in almost every part of Machine Learning and Deep Learning.


# Non-Linear Activation Functions

Modern neural network models use non-linear activation functions. They allow the model to create complex mappings between the network’s inputs and outputs, which are essential for learning and modeling complex data, such as images, video, audio, and data sets which are non-linear or have high dimensionality.

### Sigmoid Activation Function

![sigmoid](https://miro.medium.com/max/1626/0*p2_ajun9TmHxOq3r.png)

The sigmoid activation function is also called the logistic function. It is the same function used in the logistic regression classification algorithm.

The sigmoid activation function is calculated as follows:

> 1.0 / (1.0 + e^-x)

When using the Sigmoid function for hidden layers, it is a good practice to use a “Xavier Normal” or “Xavier Uniform” weight initialization (also referred to Glorot initialization, named for Xavier Glorot) and scale input data to the range 0-1 (e.g. the range of the activation function) prior to training.

**The major advantages of sigmoid function are:**

    The function is differentiable. That means, we can find the slope of the sigmoid curve at any two points.
    
    The function is monotonic.

    Smooth gradient, preventing “jumps” in output values.
    
    Output values bound between 0 and 1, normalizing the output of each neuron. Therefore, it is especially used for models where we have to predict the probability as an output. Since probability of anything exists only between the range of 0 and 1, sigmoid is the right choice.
    
    Clear predictions — For X above 2 or below -2, tends to bring the Y value (the prediction) to the edge of the curve, very close to 1 or 0. This enables clear predictions.

**Disadvantages:**

    Vanishing gradient — If we look carefully at the graph towards the ends of the function, y values react very little to the changes in x. Let’s think about what kind of problem it is! The derivative values in these regions are very small and converge to 0. This is called the vanishing gradient and the learning is minimal. if 0, not any learning! When slow learning occurs, the optimization algorithm that minimizes error can be attached to local minimum values and cannot get maximum performance from the artificial neural network model.

    Outputs not zero centered.

    Computationally expensive

### Hyperbolic Tangent Function

![tanh](https://miro.medium.com/max/1800/0*YxiIWSNgXCJuBdf0.png)

The Tanh activation function is calculated as follows:

> (e^x – e^-x) / (e^x + e^-x)

It has a structure very similar to Sigmoid function. However, this time the function is defined as (-1, + 1), it is zero-centered. The advantage over the sigmoid function is that its derivative is more steep, which means it can get more value. This means that it will be more efficient because it has a wider range for faster learning and grading.

When using the TanH function for hidden layers, it is a good practice to use a “Xavier Normal” or “Xavier Uniform” weight initialization (also referred to Glorot initialization, named for Xavier Glorot) and scale input data to the range -1 to 1 (e.g. the range of the activation function) prior to training. 

However, it has similar disadvantages as sigmoid function.


### RELU

![RELU](https://miro.medium.com/max/2100/0*M5PI4ur_mkwV0A1W.png)

Specifically, it is less susceptible to vanishing gradients that prevent deep models from being trained, although it can suffer from other problems like saturated or “dead” units.

The ReLU function is calculated as follows:

> max(0.0, x)

**Advantages**

    Computationally efficient — ReLU is valued at [0, +infinity], but what are the returns and their benefits? Let’s imagine a large neural network with too many neurons. Sigmoid and hyperbolic tangent caused almost all neurons to be activated in the same way. This means that the activation is very intensive. Some of the neurons in the network are active, and activation is infrequent, so we want an efficient computational load. We get it with ReLU. Having a value of 0 on the negative axis means that the network will run faster. The fact that the calculation load is less than the sigmoid and hyperbolic tangent functions has led to a higher preference for multi-layer networks.

    Non-linear — At first glance, it will appear to have the same characteristics as the linear function on the positive axis. But above all, ReLU is not linear in nature. In fact, a good estimator. It is also possible to converge with any other function by combinations of ReLU.

**Disadvantages**

    The Dying ReLU problem — when inputs approach zero, or are negative, the gradient of the function becomes zero, the network cannot perform backpropagation and cannot learn.

#### Leaky-ReLU Function

![L.RELU](https://miro.medium.com/max/1800/0*tf4m09jBicciHI-0.png)

![L.RELU](https://miro.medium.com/max/1710/1*V0zNSqtFkvly9c5eQ65jLA.png)

The leak helps to increase the range of the ReLU function. Usually, the value of a is 0.01 or so.

When a is not 0.01 then it is called Randomized ReLU.

**Advantages**

    Prevents dying ReLU problem — this variation of ReLU has a small positive slope in the negative area, so it does enable backpropagation, even for negative input values. This leaky value is given as a value of 0.01 if given a different value near zero, the name of the function changes randomly as Leaky ReLU. The definition range of the leaky-ReLU continues to be minus infinity. This is close to 0, but 0 with the value of the non-living gradients in the RELU lived in the negative region of learning to provide the values.
    Otherwise like ReLU

**Disadvantages**

    Results not consistent — leaky ReLU does not provide consistent predictions for negative input values.


#### Parametric ReLU

![P.RELU](https://miro.medium.com/max/1608/1*7Enzmomwlx2vAG7mwiYMUA.png)

**Advantages**

    Allows the negative slope to be learned — unlike leaky ReLU, this function provides the slope of the negative part of the function as an argument. It is, therefore, possible to perform backpropagation and learn the most appropriate value of α.
    
    Otherwise like ReLU

**Disadvantages**

    May perform differently for different problems.


### Softmax

The softmax function is calculated as follows:

> e^x / sum(e^x)

**Advantages**

    Able to handle multiple classes only one class in other activation functions — normalizes the outputs for each class between 0 and 1, and divides by their sum, giving the probability of the input value being in a specific class.

    Useful for output neurons — typically Softmax is used only for the output layer, for neural networks that need to classify inputs into multiple categories.


## Activation for Hidden Layers

There are perhaps three activation functions you may want to consider for use in hidden layers; they are:

    Rectified Linear Activation (ReLU)
    Logistic (Sigmoid)
    Hyperbolic Tangent (Tanh)

Both the sigmoid and Tanh functions can make the model more susceptible to problems during training, via the so-called vanishing gradients problem.

The activation function used in hidden layers is typically chosen based on the type of neural network architecture.

    Multilayer Perceptron (MLP): ReLU activation function.

    Convolutional Neural Network (CNN): ReLU activation function.

    Recurrent Neural Network: Tanh and/or Sigmoid activation function.


## Activation for Output Layers

There are perhaps three activation functions you may want to consider for use in the output layer; they are:

    Linear
    Logistic (Sigmoid)
    Softmax

Target values used to train a model with a linear activation function in the output layer are typically scaled prior to modeling using normalization or standardization transforms.


    Regression: One node, linear activation.

    Binary Classification: One node, sigmoid activation.

    Multiclass Classification: One node per class, softmax activation.

    Multilabel Classification: One node per class, sigmoid activation.


[Ref-1](https://xzz201920.medium.com/activation-functions-linear-non-linear-in-deep-learning-relu-sigmoid-softmax-swish-leaky-relu-a6333be712ea)
[Ref-2](https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/)