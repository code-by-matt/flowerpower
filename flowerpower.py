import sys
import numpy
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

# Load in data.
X = numpy.load("features.npy")
y = numpy.load("classes.npy")

# Split the data into a training set and a test set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Train a classifier and print a 95% confidence interval of score for each given regularization strength.
print("\nCross-Validation Scores:")
for i in range(1, len(sys.argv)):
    classifier = LogisticRegression(penalty="l2", C=float(sys.argv[i]), solver="lbfgs", multi_class="ovr")
    scores = cross_val_score(classifier, X_train, y_train, cv=5)
    print(str(i) + " %0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2))

# Calculate final performance of user-chosen regularization strength.
choice = int(input("\nEnter a row number: "))
classifier = LogisticRegression(penalty="l2", C=float(sys.argv[choice]), solver="lbfgs", multi_class="ovr")
classifier.fit(X_train, y_train)
print("Final Score: %0.3f" % (classifier.score(X_test, y_test)))
