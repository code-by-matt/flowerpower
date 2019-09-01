# Flower Power!

It's the "hello world" of machine learning: the iris data set! Here I attempt to use scikit-learn to build a classifier that predicts the species of an iris given four of its features. I document the whole process below.

## The Plan

[Andrew Ng](https://coursera.org/learn/machine-learning) taught me how to train a logistic regression classifier in Octave by writing my own implementation of gradient descent and manually choose a learning rate and a regularization parameter, but apparently all that was baby stuff. Real ML engineers train classifiers using ready-made solutions like this!

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty="l2", C="1.0", solver="lbfgs", multi_class="ovr")
classifier.fit(X, y)
```

The `LogisticRegression` constructor has many optional arguments: `penalty="l2"` specifies a certain regularization scheme (the only scheme I'm familiar with lol), `C="1.0"` sets the inverse of regularization strength (why inverse? nobody knows), `solver="lbfgs"` means that the classifier will use the [limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS), which I guess is some hella fancy version of gradient descent, and `multi_class="ovr"` means one-versus-rest.

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