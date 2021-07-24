
# Boosting

Can a set of weak learners create a single strong learner? A weak learner is defined to be a classifier which is only slightly correlated with the true classification (it can label examples better than random guessing).


# Random Forest

Random Forests grows many classification trees. To classify a new object from an input vector, put the input vector down each of the trees in the forest. Each tree gives a classification, and we say the tree "votes" for that class. The forest chooses the classification having the most votes (over all the trees in the forest).

A random forest is a meta estimator that fits a number of decision tree classifiers on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting.

Random Forest is a bagging algorithm. Bagging and Boosting are two opposite way to achieve a low error.

*We know that error can be composited from bias and variance. A too complex model has low bias but large variance, while a too simple model has low variance but large bias, both leading a high error but two different reasons. As a result, two different ways to solve the problem come into people's mind, variance reduction for a complex model, or bias reduction for a simple model, which refers to random forest and boosting.*

**Random forest reduces variance of a large number of "complex" models with low bias.** 
The underlying trees are independent parallel models. 

Boosting reduces bias of a large number of "small" models with low variance. They are "weak" models.
In boosting as the name suggests, one is learning from other which in turn boosts the learning.
The underlying elements are somehow like a "chain" or "nested" iterative model about the bias of each level. So they are not independent parallel models but each model is built based on all the former small models by weighting. That is so-called "boosting" from one by one.
The trees in boosting algorithms like GBM-Gradient Boosting machine are trained sequentially.

**An example of this iterative process is adaboost**, whereby weaker results are boosted or reweighted over many iterations to have the learner focus more on areas it got wrong, and less on those observations that were correct.

Let's say the first tree got trained and it did some predictions on the training data. Not all of these predictions would be correct. Let's say out of a total of 100 predictions, the first tree made mistake for 10 observations. Now these 10 observations would be given more weightage when building the second tree. Notice that the learning of the second tree got boosted from the learning of the first tree. Hence, the term boosting. This way, each of the trees are built sequentially over the learnings from the past trees.


A random forest, in contrast, is an ensemble bagging or averaging method that aims to reduce the variance of individual trees by randomly selecting (and thus de-correlating) many trees from the dataset, and averaging them.


[Reference](https://stats.stackexchange.com/questions/77018/is-random-forest-a-boosting-algorithm)