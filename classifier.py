import sys
import numpy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

# Load in data.
X = numpy.load("features.npy")
y = numpy.load("classes.npy")

# Split the data into a training set and a test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train a classifier for each given regularization strength and record its accuracy.
results = []
for r in sys.argv:
    classifier = LogisticRegression(penalty="l2", C=r, solver="lbfgs", multi_class="ovr")
    classifier.fit(X_train, y_train)
    scores = cross_val_score(classifier, X_test, y_test, cv=5)
    results.append("%0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))
print(results)
