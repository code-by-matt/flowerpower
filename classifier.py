import numpy
from sklearn.model_selection import train_test_split

# Load in data.
X = numpy.load("features.npy")
y = numpy.load("classes.npy")

# Split the data into a training set and a test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
