import sklearn.datasets
import sklearn.linear_model
import numpy.random
import numpy.linalg
import matplotlib.pyplot

if __name__ == "__main__":
# Load boston dataset
    boston = sklearn.datasets.load_boston()

# Split the dataset with sampleRatio
sampleRatio = 0.5
n_samples = len(boston.target)
sampleBoundary = int(n_samples * sampleRatio)

# Shuffle the whole data
shuffleIdx = list(range(n_samples))
numpy.random.shuffle(shuffleIdx)


# Make the training data
train_features = boston.data[shuffleIdx[:sampleBoundary]]
train_targets = boston.target[shuffleIdx [:sampleBoundary]]

# Make the testing data
test_features = boston.data[shuffleIdx[sampleBoundary:]]
test_targets = boston.target[shuffleIdx[sampleBoundary:]]

# Train
linearRegression = sklearn.linear_model.LinearRegression()
linearRegression.fit(train_features, train_targets)

# Predict
predict_targets = linearRegression.predict(test_features)

# Evaluation
n_test_samples = len(test_targets)
X = range(n_test_samples)
error = numpy.linalg.norm(predict_targets - test_targets, ord = 1) / n_test_samples
print ("Ordinary Least Squares (Boston) Error: %.2f" %(error))
# Draw
matplotlib.pyplot.plot(X, predict_targets, 'r--*', label = 'Predict Price')
matplotlib.pyplot.plot(X, test_targets, 'g:', label='True Price')
legend = matplotlib.pyplot.legend()
matplotlib.pyplot.title("Ordinary Least Squares (Boston)")
matplotlib.pyplot.ylabel("Price")
matplotlib.pyplot.savefig("Ordinary Least Squares (Boston).png", format='png')
matplotlib.pyplot.show()