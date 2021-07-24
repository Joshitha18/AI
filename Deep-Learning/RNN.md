Recurrent Neural Networks (RNNs) are a popular variant of artificial neural networks which work really well on sequential data types i.e. a set of data points which are arranged in a particular order such that related data points follow each other. Some examples of sequential data types are stock market prices, audio and video data, DNA sequences, sensor data, natural language text, etc.

RNNs are able to predict the next output of a sequence by taking information from the previous sequence and combining it with the input of the current sequence.

# RNN Design Architectures

## 1. Many-to-many architecture (same sequence length)

This is the basic RNN structure where the number of input sequence is the same as the number of output sequence at every time-step. An example application that uses this architecture is a text generator that predicts the next most likely word in a sentence, given the previous words.

![RNN](https://miro.medium.com/max/5760/1*w_DRBfEU0SLHOJBKfaFAGA.png)

In the image above, the text generator’s input is a sequence of words “The brown dog has four” and its output is also a sequence of predicted next words “brown dog has four legs.”


## 2. Many-to-many architecture (different sequence length)

Another variation of the many-to-many architecture is in cases where the input and output both have different sequence lengths. An example application is in machine translation tasks, where the input is a sequence of words in a source language (e.g. French) and the output is a sequence of words in a target language (e.g English). This architecture has two distinct parts; an encoder which takes in the input sentence, maps it into an internal state representation and then passes this representation to a decoder which then uses it to generate the output sentence.

![RNN](https://miro.medium.com/max/1680/1*NC9mt2tK-u_a98tP7r_a1w.png)

In this architecture the decoder can only start predicting the output sequence after the encoder has processed the complete input sequence, unlike in the same sequence length architecture which starts predicting the output sequence immediately after each input sequence


## 3. Many-to-one architecture

In this architecture the RNN has a sequence of inputs at each time step but only outputs a single value at the last time step. An example application that is a sentiment analysis task, where the objective is to classify a given input statement as having either a positive or negative sentiment.

In this architecture the RNN has a sequence of inputs at each time step but only outputs a single value at the last time step. An example application that is a sentiment analysis task, where the objective is to classify a given input statement as having either a positive or negative sentiment.


## 4. One-to-many architecture

Here the RNN takes in a single input at the first time step and outputs a sequence of values at the remaining time steps. In this architecture , some applications often also take the predicted output at each time step and feed it into the next layer as an input value. An example application for this architecture is in image captioning, here the RNN takes an image as its input and outputs a sequence of words describing what is going on in the image.

![RNN](https://miro.medium.com/max/1680/1*BYaRjcqtcmyOIEUr1FICjw.png)

In the above, the input is an emoji of a woman running and the output is a sequence of predicted words that describe the image “woman in blue vest running”.


# Forward propagation in single RNN Cell

![Forward propagation RNN](https://miro.medium.com/max/1680/1*L1Ws8Hb3rx4Oq_rfOz0ZmA.gif)

[Read here](https://medium.com/learn-love-ai/step-by-step-walkthrough-of-rnn-training-part-i-7aee5672dea3)

Forward propagation through time is simply running the steps above in the whole recurrent network rather than in just a single time-step RNN cell. It begins with the initialization of a hidden state a⟨0⟩ , sharing of the weights and bias vectors W_xh, W_ah, W_ao, b_h, b_o across all time steps t = 1 to T, and repeating the steps above in each time step.


# Back propagation in single RNN Cell

The goal of back propagation in the RNN is to compute the partial derivatives of the weight matrices (W_xh, W_ah, W_ao) and bias vectors (b_h, b_o) with respect to the final loss L .

![Back Propogation RNN](https://miro.medium.com/max/1680/1*Zc13qWUltAc_H0fASvTibg.gif)

[Read here](https://medium.com/learn-love-ai/step-by-step-walkthrough-of-rnn-training-part-ii-7141084d274b)

Just like during forward propagation, BPTT is also just running the above steps backwards through the whole unrolled recurrent network.

The major difference here is that to update the weights and biases, we have to calculate the sum of each partial derivative 𝜕W_ao, 𝜕b_o, 𝜕W_ah, 𝜕W_xh, 𝜕b_h, at every time step t, because these parameters are shared across during forward propagation.


# Vanishing & Exploding gradient

## Why Gradients Explode or Vanish

Say, in many-to-many RNN architecture for text generation, lets assume the input sequence to the network is a 20 word sentence: “I grew up in France,…….. I speak French fluently.

We can see from the example above that for the RNN to predict the word “French” which comes at the end of the sequence, it would need information from the word “France”, which occurs further back at the beginning of the sentence.
This kind of dependence between sequence data is called long-term dependencies because the distance between the relevant information “France” and the point where it is needed to make a prediction “French” is very wide. Unfortunately, in practice as this distance becomes wider, RNNs have a hard time learning these dependencies because it encounters either a vanishing or exploding gradient problem.

**These problems arise during training of a deep network when the gradients are being propagated back in time all the way to the initial layer. The gradients coming from the deeper layers have to go through continuous matrix multiplications because of the the chain rule, and as they approach the earlier layers, if they have small values (<1), they shrink exponentially until they vanish and make it impossible for the model to learn , this is the vanishing gradient problem. While on the other hand if they have large values (>1) they get larger and eventually blow up and crash the model, this is the exploding gradient problem**


# Dealing with Exploding Gradients

When gradients explode, the gradients could become NaN because of the numerical overflow or we might see irregular oscillations in training cost when we plot the learning curve. A solution to fix this is to apply gradient clipping; which places a predefined threshold on the gradients to prevent it from getting too large, and by doing this it doesn’t change the direction of the gradients it only change its length.

## Gradient Clipping

There are many ways to compute gradient clipping, but a common one is to rescale gradients so that their norm is at most a particular value. With gradient clipping, pre-determined gradient threshold be introduced, and  then gradients norms that exceed this threshold are scaled down to match the norm.  This prevents any gradient to have norm greater than the threshold and thus the gradients are clipped.  There is an introduced bias in the resulting values from the gradient, but gradient clipping can keep things stable. 

Two types of gradient clipping can be used: gradient norm scaling and gradient value clipping.

![Gradient clipping](https://miro.medium.com/max/1890/1*vLFINWklJ0BtYtgzwK223g.png)

### Gradient Norm Scaling

Gradient norm scaling involves changing the derivatives of the loss function to have a given vector norm when the L2 vector norm (sum of the squared values) of the gradient vector exceeds a threshold value.

For example, we could specify a norm of 1.0, meaning that if the vector norm for a gradient exceeds 1.0, then the values in the vector will be rescaled so that the norm of the vector equals 1.0.

### Gradient Value Clipping

Gradient value clipping involves clipping the derivatives of the loss function to have a given value if a gradient value is less than a negative threshold or more than the positive threshold.

For example, we could specify a norm of 0.5, meaning that if a gradient value was less than -0.5, it is set to -0.5 and if it is more than 0.5, then it will be set to 0.5.


# Dealing with Vanishing Gradients

One simple solution for dealing with vanishing gradient is the identity RNN architecture; where the network weights are initialized to the identity matrix and the activation functions are all set to ReLU and this ends up encouraging the network computations to stay close to the identity function. This works well because when the error derivatives are being propagated backwards through time, they remain constants of either 0 or 1, hence aren’t likely to suffer from vanishing gradients.


# LSTM

An even more popular and widely used solution is the Long Short-Term Memory architecture (LSTM); a variant of the regular recurrent network which was designed to make it easy to capture long-term dependencies in sequence data. The standard RNN operates in such a way that the hidden state activation are influenced by the other local activations closest to them, which corresponds to a “short-term memory”, while the network weights are influenced by the computations that take place over entire long sequences, which corresponds to a “long-term memory”. Hence the RNN was redesigned so that it has an activation state that can also act like weights and preserve information over long distances, hence the name “Long Short-Term Memory”.

![LSTM](https://miro.medium.com/max/1400/1*P6R8B-qP0Oct-R9zkcAfcA.gif)

The LSTM, just like in a standard RNN receives its input from the current time-step input x<t> and from the previous time-step hidden state activation a<t-1> . 

The main structural differences between the two units are:
1. the introduction of a memory cell state c<t>,
2. introduction of three sigmoid gates (forget gate σf<t>,update gate σu<t>,output gateσo<t> ),
3. and the ability to remove or add information to the memory cell state.

[Read more about forward propogation of LSTM here](https://medium.com/learn-love-ai/and-of-course-lstm-part-i-b226880fb287)

