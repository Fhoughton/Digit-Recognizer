from keras import utils

from keras.layers import Dense, Reshape, Conv2D, AveragePooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop

import numpy as np
import pandas as pd

data_size = 28
colour_threshold = 100 / 255

file = open("train.csv")
train_df1 = pd.read_csv(file)

file = open("mnist_test.csv")
train_df2 = pd.read_csv(file)

file = open("mnist_train.csv")
train_df3 = pd.read_csv(file)

y_train1 = np.array(train_df1.iloc[:, 0])
y_train2 = np.array(train_df2.iloc[:, 0])
y_train3 = np.array(train_df3.iloc[:, 0])

x_train1 = np.array(train_df1.iloc[:, 1:])
x_train2 = np.array(train_df2.iloc[:, 1:])
x_train3 = np.array(train_df3.iloc[:, 1:])

x_train = np.row_stack((x_train1,x_train2, x_train3))
y_train = np.append(np.append(y_train1,y_train2), y_train3)

file = open("test.csv")
test_df = pd.read_csv(file)
x_test = np.array(test_df)

n_samples_train = x_train.shape[0]
n_samples_test = x_test.shape[0]
        
def inttofloat(x):
    x = x / 255
    return x

def output(prediction, model_name):
    df_predict = {"ImageId":range(1, n_samples_test+1), "Label":prediction}
    df_predict = pd.DataFrame(df_predict)
    df_predict.to_csv("Submission.csv", index = False)
    
x_train = inttofloat(x_train)
x_test = inttofloat(x_test)
    
model_name = "CNN"
model = Sequential()
y_train = utils.to_categorical(y_train, num_classes=10)

model.add(Reshape(target_shape=(1, 28, 28), input_shape=(784,)))
model.add(Conv2D(kernel_size=(3, 3), filters=32, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Conv2D(kernel_size=(3, 3), filters=64, padding="same", data_format="channels_first", kernel_initializer="uniform", use_bias=False))
model.add(AveragePooling2D(pool_size=(2, 2), data_format="channels_first"))
model.add(Flatten())
model.add(Dense(output_dim=1000, activation='relu'))
model.add(Dense(output_dim=100, activation='relu'))
model.add(Dense(output_dim=10, activation='softmax'))

rmsprop = RMSprop(lr=0.0005)
model.compile(loss='binary_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=60, batch_size=64)
prediction = model.predict_classes(x_test)

output(prediction, model_name)