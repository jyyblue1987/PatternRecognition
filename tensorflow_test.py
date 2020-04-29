from keras.utils import to_categorical

from keras import models
from keras import layers

import pandas as pd

df = pd.read_csv(r'D_Train1.csv')
train_data = df.to_numpy()

X_Train = train_data[:, 1:]
Y_Train = train_data[:, [0]]
Y_Train = Y_Train.ravel()
Y_Train = Y_Train - 1

df = pd.read_csv(r'D_Test1.csv')

test_data = df.to_numpy()

X_Test = test_data[:, 1:]
Y_Test = test_data[:, [0]]
Y_Test = Y_Test.ravel()
Y_Test = Y_Test - 1

X_Train = X_Train.astype('float32') / 100
X_Test = X_Test.astype('float32') / 100
train_labels = to_categorical(Y_Train)
test_labels = to_categorical(Y_Test)

dim = len(X_Train[0])
label_count = len(train_labels[0])

network1 = models.Sequential()
network1.add(layers.Dense(32, activation='relu', input_shape=(dim,)))
network1.add(layers.Dense(label_count, activation='softmax'))

network1.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])



network1.fit(X_Train, train_labels, epochs=50, batch_size=32)
test_loss, test_acc = network1.evaluate(X_Test, test_labels)
print('test_acc:', test_acc)