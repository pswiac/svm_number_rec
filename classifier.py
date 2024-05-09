# Python script for training an SVM classifier for handwritten digit recognition from MNIST dataset
# and saving it to a file

import joblib
import pandas as pd
from sklearn import svm, metrics

df = pd.read_csv("train.csv")
dft = pd.read_csv("test.csv")

train = df.values
test = dft.values

train_data = train[:, 1:]
train_labels = train[:, 0]

test_data = test[:, 1:]
test_labels = test[:, 0]

train_data = train_data / 255
test_data = test_data / 255

classifier = svm.SVC(C=200, kernel='rbf', gamma=0.01, probability=False)
classifier.fit(train_data, train_labels)

predicted = classifier.predict(test_data)

joblib.dump(classifier, 'svm_classifier.pkl')