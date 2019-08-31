# Flower Power!
It's the "hello world" of machine learning: the iris data set! Here I attempt to use scikit-learn to build a classifier that predicts the species of an iris given four of its features. I document the whole process below.

## Reading in Data
I grabbed the `iris.data` file from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php), where it happens to be the site's most popular data set. The first thing to do was to package and save the data into two big numpy arrays. I stored the 150-by-4 feature array in `features.npy`, and it's simply the first four columns of `iris.data`.
```python
import numpy
X = numpy.loadtxt("iris.data", delimiter=",", usecols=(0,1,2,3))
numpy.save("features", X)
```
It has yet to be decided how to form the class array...
