# Flower Power!

I recently finished Andrew Ng's famous [machine learning course](https://coursera.org/learn/machine-learning). I gained a fundamental understanding of a variety of ML topics, but I left the course thinking, "Wait, so how do people actually *use* these things?" To answer this question, I decided to try my hand at running logistic regression on the ever-so-popular<sup>[1]</sup> [iris data set](https://en.wikipedia.org/wiki/Iris_flower_data_set) using scikit-learn, a free Python machine learning library. Read on to find out what I learned!

## Exploring the Docs

I initially thought that this project was going to be pretty similar to the assignments in Andrew's course where I implemented various machine learning algorithms in Octave. I would ponder the algorithm in an abstract sense, then try to efficiently translate it into code. Making sure my implementation worked properly was a slow and fiddly process, riddled with missing semicolons and botched matrix operations. You really had to know what your code was doing. Turns out, with scikit-learn, you don't need to know what you're doing at all!

Just kidding (kinda). Going through that course was definitely a valuable experience, but a quick tour of the scikit-learn documentation made me realize that *all the hard stuff was already done for me*. Example: My work for Andrew's one-vs-all logistic regression assignment was a jank collection of scripts written in TextEdit because I was too lazy to find a better editor.

![My work for Andrew Ng's logistic regression assignment.](jankness.png)

I did the same thing with scikit-learn in two lines.

```python
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", multi_class="ovr")
```

The `LogisticRegression` constructor has tons of optional arguments, of which I chose four: `penalty="l2"` specified a certain regularization scheme,<sup>[2]</sup> `C=1.0` set the inverse of regularization strength,<sup>[3]</sup> `solver="lbfgs"` meant that the classifier used the [limited-memory Broyden-Fletcher-Goldfarb-Shanno algorithm](https://en.wikipedia.org/wiki/Limited-memory_BFGS), which seems to be some hella fancy version of gradient descent, and `multi_class="ovr"` meant one-versus-rest. These two lines were all I needed to create a fully-functional logistic regression classifier. Fitting the classifier is as easy as calling `classifier.fit()`, and making predictions on data is as easy as calling `classifier.predict()`. All the fiddly implementation details had already been figured out and hidden away!

It became clear that I didn't need to reinvent the wheel–I simply needed to choose a few from a massive library of ready-made wheels.

## Reading in the Data

Before the wheel shopping could begin, however, I needed to gather my data and make it compatible with scikit-learn. I started with the `iris.data` file from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). The file contained 150 rows of data, where each row consisted of four flower features (sepal length, sepal width, petal length, petal width) and a class (one of three iris species).

```
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
...
5.5,2.6,4.4,1.2,Iris-versicolor
6.1,3.0,4.6,1.4,Iris-versicolor
...
6.7,2.5,5.8,1.8,Iris-virginica
7.2,3.6,6.1,2.5,Iris-virginica
...
```

I saved the first four columns of the data in a 150-by-4 numpy array called `features.npy`, and I stored the fifth column in a 150-by-1 numpy array called `classes.npy`.

```python
import numpy
X = numpy.loadtxt("iris.data", delimiter=",", usecols=(0,1,2,3))
y = numpy.loadtxt("iris.data", delimiter=",", usecols=(4), dtype="str")
numpy.save("features", X)
numpy.save("classes", y)
```

Then, while skimming the docs, I discovered that the iris data set is *literally built into scikit-learn*.

```python
from sklearn import datasets
iris = datasets.load_iris()
```

Sigh. Moving on then.

## Training Methodology

As is tradition, I wanted to train my classifier on one portion of the data and test its performance on another. This is because I wanted to see how well my classifier captures the underlying behavior of irises at large, not the idosyncracies of data I trained with. If I trained and tested on the same data, there would be no way to tell if my classifier was *overfitted*, i.e., capturing a lot of idosyncracies and obscuring the correct underlying behavior. The `train_test_split` function was a super easy way to randomly split up the data into a training set and a test set using any proportion I wanted. In this case, I saved 30% of the data for testing.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
```

Keen to try out more techniques from Andrew's course, I wanted to run the classifier with a bunch of different `C` values and see which one is the best. But again, I couldn't check the performance of a `C` value by testing on the same data I trained with. The way this was dealt with in the course was by making a *cross-validation set*, to be used only for comparing different regularization strengths. I found a better way in scikit-learn's `cross_val_score`, which does a fancy thing called [*k-fold validation*](https://scikit-learn.org/stable/modules/cross_validation.html).

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(classifier, X_train, y_train, cv=5)
interval = "%0.3f (+/- %0.3f)" % (scores.mean(), scores.std() * 2)
```

By setting `cv=5`, I told scikit-learn to train the classifier five times on five different training/cross-validation splits, or "folds". The accuracy acheived on each fold was recorded in `scores` as floats between 0 and 1, with 1 meaning 100% accurate. (The way the folds are determined and the exact meaning of "accurate" depend on a variety of factors; see the above link.) From `scores` I then constructed a 95% confidence interval for accuracy.

## Training and Results Analysis

Once I found the pieces I needed for my plan, it was pretty quick to throw them together into a script called `flowerpower.py`. The first part of the script takes any number of `C` values as command-line arguments and computes their cross-validation scores on the training set. The second part calculates a final score on the testing set using a `C` value that you choose. Here's one example output.

```
$ python flowerpower.py 0.01 100 1000000

Cross-Validation Scores:
1 0.731 (+/- 0.117)
2 0.942 (+/- 0.042)
3 0.952 (+/- 0.063)

Enter a row number: 2
Final Score: 0.978
```

At `C=0.01`, regularization was too strong and caused underfitting. With higher `C` values, the classifier performed really well, and I never observed a significant drop in performance by increasing `C`. This means that the data conforms to a simple pattern and is not very susceptible to overfitting, which was kind of a bummer because I might as well have not used regularization at all.

## Conclusion

Freeman Dyson said, "A good scientist is a person with original ideas. A good engineer is a person who makes a design that works with as few original ideas as possible." I approached this project trying to be a good scientist, but found out that I really needed to be a good engineer. I spent more time reading the docs, choosing tools to use, and writing up this report than on actually coding. And even though I didn't learn any new machine learning theory, there's something satisfying solving a problem simply and clearly, in as few steps as possible.<sup>[4]</sup>

I'm by no means a master of scikit-learn now. There's so much more out there to know, and there are probably better ways to do what I did. Nevertheless, I hope that my experience inspires you to go out and build that thing you've always wanted to build. Chances are, somebody's already made some really nice wheels for you :)

---

1. It's so popular, it tops Wikipedia's [list of classic data sets](https://en.wikipedia.org/wiki/Data_set#Classic_data_sets)–because of course such a thing exists.

2. L2 regularization is what I saw in Andrew's course, so I chose it out of familiarity. The precise difference in behavior between L1 regularization and L2 regularization remains a mystery to me.

3. Why inverse? Nobody knows.

4. I guess that's why some people like golf.
