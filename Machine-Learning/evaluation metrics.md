
# Evaluating classification models

### Confusion Matrix

Let's make the following definitions:

"Wolf" is a positive class.
"No wolf" is a negative class.

We can summarize our "wolf-prediction" model using a 2x2 confusion matrix that depicts all four possible outcomes:

![cm](https://eastceylon.com/images/2021/07/24/Screenshot-2021-07-24-at-21-54-11-Classification-True-vs-False-and-Positive-vs-Negative.png)

A true positive is an outcome where the model correctly predicts the positive class. Similarly, a true negative is an outcome where the model correctly predicts the negative class.

A false positive is an outcome where the model incorrectly predicts the positive class. And a false negative is an outcome where the model incorrectly predicts the negative class.

## Accuracy

Accuracy is one metric for evaluating classification models. Informally, accuracy is the fraction of predictions our model got right. Formally, accuracy has the following definition:

**Accuracy = # correct predictions/# total predictions**

For binary classification, accuracy can also be calculated in terms of positives and negatives as follows:

**Accuracy = TP + TN / ( TP + TN + FP + FN )**

Where TP = True Positives, TN = True Negatives, FP = False Positives, and FN = False Negatives.

*Accuracy alone doesn't tell the full story when you're working with a class-imbalanced data set, like the one, where there is a significant disparity between the number of positive and negative labels.*


## Precision and Recall

Precision attempts to answer the following question:

    What proportion of positive identifications was actually correct?

Precision is defined as follows:

Precision = TP/(TP + FP)


Recall attempts to answer the following question:

    What proportion of actual positives was identified correctly?

Mathematically, recall is defined as follows:

Recall = TP/(TP + FN)


### Precision and Recall: A Tug of War

To fully evaluate the effectiveness of a model, you must examine both precision and recall. Unfortunately, precision and recall are often in tension. That is, improving precision typically reduces recall and vice versa. 

## The Role of the F1-Score

We need a tradeoff between Precision and Recall. We first need to decide which is more important for our classification problem.

There are also a lot of situations where both precision and recall are equally important. 

In such cases, we use something called F1-score. F1-score is the Harmonic mean of the Precision and Recall:

![img](https://cdn.analyticsvidhya.com/wp-content/uploads/2019/09/f1score.png)

This is easier to work with since now, instead of balancing precision and recall, we can just aim for a good F1-score and that would be indicative of a good Precision and a good Recall value as well.

## ROC curve

An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots two parameters:

- True Positive Rate (y-axis)
- False Positive Rate (x-axis)

True Positive Rate (TPR) is a synonym for recall and is therefore defined as follows:

TPR = TP/(TP + FN)

False Positive Rate (FPR) is defined as follows:

FPR = FP/(FP + TN)

An ROC curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives. 

To compute the points in an ROC curve, we could evaluate a logistic regression model many times with different classification thresholds, but this would be inefficient. Fortunately, there's an efficient, sorting-based algorithm that can provide this information for us, called AUC.

### AUC: Area Under the ROC Curve

![auc](https://miro.medium.com/max/1083/1*pk05QGzoWhCgRiiFbz-oKQ.png)


AUC stands for "Area under the ROC Curve." That is, AUC measures the entire two-dimensional area underneath the entire ROC curve (think integral calculus) from (0,0) to (1,1).

AUC ranges in value from 0 to 1. A model whose predictions are 100% wrong has an AUC of 0.0; one whose predictions are 100% correct has an AUC of 1.0.


AUC is desirable for the following two reasons:

    AUC is scale-invariant. It measures how well predictions are ranked, rather than their absolute values.
    AUC is classification-threshold-invariant. It measures the quality of the model's predictions irrespective of what classification threshold is chosen.

However, both these reasons come with caveats, which may limit the usefulness of AUC in certain use cases:

    Scale invariance is not always desirable. For example, sometimes we really do need well calibrated probability outputs, and AUC won’t tell us about that.

    Classification-threshold invariance is not always desirable. In cases where there are wide disparities in the cost of false negatives vs. false positives, it may be critical to minimize one type of classification error. For example, when doing email spam detection, you likely want to prioritize minimizing false positives (even if that results in a significant increase of false negatives). AUC isn't a useful metric for this type of optimization.
