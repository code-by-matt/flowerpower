# Flower Power!

It's the "hello world" of machine learning: the iris data set! Here I attempt to use scikit-learn to build a classifier that predicts the species of an iris given four of its features. I document the whole process below.

## The Plan

I wanna use linear regression with no regularization, for the sake of simplicity. [Andrew Ng](https://coursera.org/learn/machine-learning) taught me how to implement gradient descent and manually choose a learning rate and a regularization level, but apparently that was all baby stuff. Training a classifier with scikit-learn only takes three lines.

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty="none", solver="lbfgs", multi_class="ovr")
classifier.fit(X, y)
```

The `LogisticRegression` constructor takes a lot of optional arguments, but these are the ones I've tentatively concluded are relevant to me: `penalty="none"` means no regularization, `solver="lbfgs"` means that the classifier will use the [limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS), which I guess is some hella fancy version of gradient descent, and `multi_class="ovr"` means one-versus-rest.

It remains to be seen how to evaluate classifiers by plotting learning curves or other methods...

## Reading in Data

I grabbed the `iris.data` file from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), where it happens to be the site's most popular data set. (I later realized that scikit-learn comes pre-packaged with the iris data set, but this is still a good exercise in data-wrangling for me.) The first thing to do was to package and save the data into two big numpy arrays. I stored the 150-by-4 feature array in `features.npy`, and it's simply the first four columns of `iris.data`.

```python
import numpy
X = numpy.loadtxt("iris.data", delimiter=",", usecols=(0,1,2,3))
numpy.save("features", X)
```

It has yet to be decided how to form the class array...
