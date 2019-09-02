# Flower Power!

It's the "hello world" of machine learning: the iris data set! Here I attempt to use scikit-learn to build a classifier that predicts the species of an iris given four of its features. I document the whole process below.

## The Plan

[Andrew Ng](https://coursera.org/learn/machine-learning) taught me how to train a logistic regression classifier in Octave by writing my own implementation of gradient descent and manually choose a learning rate and a regularization parameter, but apparently all that was baby stuff. Real ML engineers train classifiers using ready-made solutions like this!

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", multi_class="ovr")
classifier.fit(X, y)
```

The `LogisticRegression` constructor has many optional arguments: `penalty="l2"` specifies a certain regularization scheme (the only scheme I'm familiar with lol), `C=1.0` sets the inverse of regularization strength (why inverse? nobody knows), `solver="lbfgs"` means that the classifier will use the [limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS), which I guess is some hella fancy version of gradient descent, and `multi_class="ovr"` means one-versus-rest.

Since the hard work of implementing logistic regression had been done for us, all we have to worry about is what comes before and after. We start with what comes before, which is prepping our data.

## Reading in Data

I grabbed the `iris.data` file from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), where it happens to be the site's most popular data set. (I later realized that scikit-learn comes pre-packaged with the iris data set, but this is still a good exercise in data-wrangling for me.) The first thing to do was to package and save the data into two big numpy arrays. I stored the first four columns of the data in a 150-by-4 array called `features.npy`, and I stored the fifth column in a 150-by-1 array called `classes.npy`.

```python
import numpy
X = numpy.loadtxt("iris.data", delimiter=",", usecols=(0,1,2,3))
y = numpy.loadtxt("iris.data", delimiter=",", usecols=(4), dtype="str")
numpy.save("features", X)
numpy.save("classes", y)
```

The UCI file stored the iris classes as strings (full species names), which is why I needed `dtype="str"`. 

## Partitioning the Data

We have 150 samples to work with, a portion of which we need to NOT use in training our classifier, but instead use to test our classifier after it's been trained. This is common practice, because we want to see how well our classifier captures the underlying behavior of irises at large, not the idosyncracies of data we trained on. If we trained and tested on the same data, there would be no way to tell if our accuracy is derived from the idosyncracies or correct underlying behavior. The `train_test_split` function gives us a super easy way to randomly split up our data into a training set and a test set using any proportion we want.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

In this case, we save 30% of our data for testing.

## Cross-Validation

We want to pick a good regularization strength. Andrew taught us that the way to do this is to run our classifier with a variety of strengths and then check which one is the best. But again, we can't check the performance of a particular strength by using the same data we trained with. The way this was dealt with in the course was by making a cross-validation set, to be used only for comparing different regularization strengths. The obvious drawback of splitting up the data into three parts (training, cross-validation, testing) is that you have even less data to train with. A better way is to use scikit-learn's `cross_val_score`, which does a fancy thing called [k-fold validation](https://scikit-learn.org/stable/modules/cross_validation.html).

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=5)
```

By setting `cv=5`, we tell the computer to train our classifier five times on five different training/cross-validation splits, or "folds". The accuracy acheived on each fold is recorded in `scores` as floats between 0 and 1, with 1 meaning 100% accurate. The way the folds are determined and the exact meaning of "accurate" depend on a variety of factors; see the above link.