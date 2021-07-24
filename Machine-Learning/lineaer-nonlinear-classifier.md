## How do we decide if a classifier is linear or non linear ?

A classifier is linear if its decision boundary on the feature space is a linear function: positive and negative examples are separated by an hyperplane.

This is what a SVM does by definition without the use of the kernel trick.

Logistic regression uses linear decision boundaries. Imagine you trained a logistic regression and obtained the coefficients βi. You might want to classify a test record x=(x1,…,xk) if P(x)>0.5. 
P(x)>0.5 defines a hyperplane on the feature space which separates positive from negative examples.

With kNN you don't have an hyperplane in general. Imagine some dense region of positive points. The decision boundary to classify test instances around those points will look like a curve - not a hyperplane.

# SVM

SVM is a Supervised Machine Learning Algorithm which solves both the Regression problems and Classification problems. SVM finds a hyperplane that segregates the labeled dataset(Supervised Machine Learning) into two classes.

The **hyperplane** is a line which linearly divides and classifies the data. When we add the new testing data, whatever side of the hyperplane it goes will eventually decide the class that we assign to it.

To select the right hyperplane we choose hyperplane which has a maximum possible margin between the hyperplane and any point within the dataset. Therefore this gives a fair chance to classify new data correctly.

![linearly seperable data](https://www.aitude.com/wp-content/uploads/2020/02/SVM2-1.jpg)
linearly seperable data

![non-linearly seperable data](https://www.aitude.com/wp-content/uploads/2020/02/SVM3-1-1.jpg)
non-linearly seperable data

When we can easily separate data with hyperplane by drawing a straight line is Linear SVM. When we cannot separate data with a straight line we use Non – Linear SVM. In this, we have Kernel functions. They transform non-linear spaces into linear spaces. It transforms data into another dimension so that the data can be classified.


References:
[SVM](https://www.aitude.com/svm-difference-between-linear-and-non-linear-models/)
[linear/nonlinear classifier](https://stats.stackexchange.com/questions/178522/why-knn-is-a-non-linear-classifier)