import sklearn.linear_model
import numpy.random
import matplotlib.pyplot
import sklearn.datasets

if __name__ == "__main__":
# Load iris dataset
    iris = sklearn.datasets.load_iris()

# Split the dataset with sampleRatio
sampleRatio = 0.5
n_samples = len(iris.target)
sampleBoundary = int(n_samples * sampleRatio)

# Shuffle the whole data
shuffleIdx =list( range(n_samples))
numpy.random.shuffle(shuffleIdx)


# Make the training data
train_features = iris.data[shuffleIdx[:sampleBoundary]]
train_targets = iris.target[shuffleIdx [:sampleBoundary]]

# Make the testing data
test_features = iris.data[shuffleIdx[sampleBoundary:]]
test_targets = iris.target[shuffleIdx[sampleBoundary:]]

# Train
logisticRegression = sklearn.linear_model.LogisticRegression()
logisticRegression.fit(train_features, train_targets)
# Predict
predict_targets = logisticRegression.predict(test_features)

# Evaluation
n_test_samples = len(test_targets)
X = range(n_test_samples)
correctNum = 0
for i in X:
    if predict_targets[i] == test_targets[i]:
        correctNum += 1
    accuracy = correctNum * 1.0 / n_test_samples
print ("Logistic Regression (Iris) Accuracy: %.2f" %(accuracy) )
# Draw

matplotlib.pyplot.subplot(2, 1, 1)
matplotlib.pyplot.title("Logistic Regression (Iris)")
matplotlib.pyplot.plot(X, predict_targets, 'ro-', label = 'Predict Labels')
matplotlib.pyplot.ylabel("Predict Class")
legend = matplotlib.pyplot.legend()

matplotlib.pyplot.subplot(2, 1, 2)
matplotlib.pyplot.plot(X, test_targets, 'g+-', label='True Labels')
legend = matplotlib.pyplot.legend()

matplotlib.pyplot.ylabel("True Class")
matplotlib.pyplot.savefig("Logistic Regression (Iris).png", format='png')
matplotlib.pyplot.show()

