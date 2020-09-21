import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

from tensorflow.python.keras.utils.metrics_utils import ConfusionMatrix


# Setting random seeds to keep everything deterministic.
random.seed(1618)
np.random.seed(1618)
# tf.set_random_seed(1618)   # Uncomment for TF1.
tf.random.set_seed(1618)

# Disable some troublesome logging.
# tf.logging.set_verbosity(tf.logging.ERROR)   # Uncomment for TF1.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Information on dataset.
NUM_CLASSES = 10
IMAGE_SIZE = 784

# Use these to set the algorithm to use.
# ALGORITHM = "guesser"
ALGORITHM = "custom_net"
# ALGORITHM = "tf_net"


class NeuralNetwork_2Layer:
    def __init__(self, inputSize, outputSize, neuronsPerLayer, learningRate=0.1):
        self.inputSize = inputSize
        self.outputSize = outputSize
        self.neuronsPerLayer = neuronsPerLayer
        self.lr = learningRate
        self.W1 = np.random.randn(self.inputSize, self.neuronsPerLayer)
        self.W2 = np.random.randn(self.neuronsPerLayer, self.outputSize)

    # Activation function.
    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Activation prime function.
    def __sigmoidDerivative(self, x):
        sigmoid = self.__sigmoid(x)
        return sigmoid * (1 - sigmoid)

    # Batch generator for mini-batches. Not randomized.
    def __batchGenerator(self, l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    # Training with backpropagation.
    def train(self, xVals, yVals, epochs=100000, minibatches=True, mbs=100):
        for epoch in range(epochs):
            if minibatches:
                for xBatch, yBatch in zip(self.__batchGenerator(xVals, mbs), self.__batchGenerator(yVals, mbs)):
                    self.__train_batch(xBatch, yBatch)
            else:
                self.__train_batch(xVals, yVals)

    def __train_batch(self, xVals, yVals):
        # TODO: Implement backprop. allow minibatches. mbs should specify the size of each minibatch.
        gradient_w1 = np.zeros((self.inputSize, self.neuronsPerLayer))
        gradient_w2 = np.zeros((self.neuronsPerLayer, self.outputSize))

        for xVal, yVal in zip(xVals, yVals):
            layer1, layer2 = self.__forward(xVal.flatten())
            error = (layer2 - yVal) * self.__sigmoidDerivative(layer2)
            gradient_w2 += self.W2 * error
            for index in range(len(layer1)):
                layer1[index] = self.__sigmoidDerivative(layer1[index])
            gradient_w1 += np.dot(self.W1 * self.W2.transpose()) * layer1 * error

        self.W1 -= gradient_w1
        self.W2 -= gradient_w2

    # Forward pass.
    def __forward(self, input):
        layer1 = self.__sigmoid(np.dot(input, self.W1))
        layer2 = self.__sigmoid(np.dot(layer1, self.W2))
        return layer1, layer2

    # Predict.
    def predict(self, xVals):
        _, layer2 = self.__forward(xVals)
        return layer2


# Classifier that just guesses the class label.
def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


# =========================<Pipeline Functions>==================================


def getRawData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def preprocessData(raw):
    (
        (xTrain, yTrain),
        (xTest, yTest),
    ) = raw
    xTrain = xTrain.astype(np.float) / 255.0
    xTest = xTest.astype(np.float) / 255.0
    yTrain = to_categorical(yTrain, NUM_CLASSES)
    yTest = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrain.shape))
    print("New shape of xTest dataset: %s." % str(xTest.shape))
    print("New shape of yTrain dataset: %s." % str(yTrain.shape))
    print("New shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))


def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None  # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "custom_net":
        print("Building and training Custom_NN.")
        print(
            "Not yet implemented."
        )
        network = NeuralNetwork_2Layer(IMAGE_SIZE, NUM_CLASSES, 256)
        network.train(xTrain, yTrain, epochs=10)
        return network
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(xTrain, yTrain, epochs=8)
        return model
    else:
        raise ValueError("Algorithm not recognized.")


def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "custom_net":
        print("Testing Custom_NN.")
        return model.predict(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        return model.predict(data)
    else:
        raise ValueError("Algorithm not recognized.")

def evalResults(data, preds):
    xTest, yTest = data

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    failures = np.nonzero(np.round(preds) - yTest)
    for case, failed_output in zip(failures[0], failures[1]):
        expected = yTest[case][failed_output]
        if expected == 1:
            false_negative += 1
        else:
            false_positive += 1
    true_positive = len(yTest) - false_negative
    true_negative = len(yTest) * (NUM_CLASSES - 1) - false_positive

    confusion_matrix = np.array([
        [true_negative, false_positive],
        [false_negative, true_positive]
    ])

    correct_predictions = confusion_matrix[0][0] + confusion_matrix[1][1]
    incorrect_predictions = confusion_matrix[0][1] + confusion_matrix[1][0]
    acc = correct_predictions / (correct_predictions + incorrect_predictions)

    print("Classifier algorithm: %s" % ALGORITHM)
    print("F1-Score confusion matrix:\n%s" % confusion_matrix)
    print("Classifier accuracy: %f%%" % (acc * 100))
    print()


# =========================<Main>================================================


def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)


if __name__ == "__main__":
    main()
