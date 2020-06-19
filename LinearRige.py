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

# Train with Cross Validation
ridgeRegression = sklearn.linear_model.RidgeCV(alphas=[0.01, 0.05, 0.1, 0.5, 1.0, 10.0])
#这个地方使用RidgeCV 直接交叉验证出我需要试验的几个惩罚因子，它会帮我选择这些里面在集内测试表现最优的一个参数。后面的输出选择了0.1。

ridgeRegression.fit(train_features, train_targets)
print ("Alpha = ", ridgeRegression.alpha_ )
# Predict
predict_targets = ridgeRegression.predict(test_features)

# Evaluation
n_test_samples = len(test_targets)
X = range(n_test_samples)
error = numpy.linalg.norm(predict_targets - test_targets, ord = 1) / n_test_samples
print ("Ridge Regression (Boston) Error: %.2f" %(error) )
# Draw

matplotlib.pyplot.plot(X, predict_targets, 'r--*', label = 'Predict Price')
matplotlib.pyplot.plot(X, test_targets, 'g:', label='True Price')
legend = matplotlib.pyplot.legend()
matplotlib.pyplot.title("Ridge Regression (Boston)")
matplotlib.pyplot.ylabel("Price (1000 U.S.D)")
matplotlib.pyplot.savefig("Ridge Regression (Boston).png", format='png')
matplotlib.pyplot.show()
