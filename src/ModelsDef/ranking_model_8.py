from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv3D, MaxPooling3D, BatchNormalization, Activation, Flatten
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint

from batchRankingLoss import brLoss
from evaluationMetric import evalLoss
from statisticsCoefficients import Pearson, Spearman, Kendall
from dataParser import loadData, Protein

import numpy as np

batch_size = 9
decoys = 1
max_epoch = 50
beta1 = 0.9
beta2 = 0.999
epsilon = 1E-8
learning_rate = 0.0001
weight_decay = 0.0
epochs = 10

model = Sequential()

#Custom model for loss testing
model.add(Conv3D(16, input_shape=(20,20,20,11), data_format='channels_last', kernel_size=(3,3,3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2)))

model.add(Conv3D(32, kernel_size=(3,3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2)))

model.add(Conv3D(32, kernel_size=(1,1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv3D(64, kernel_size=(1,1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2,2,2), strides=(2,2,2)))

model.add(Conv3D(128, kernel_size=(1,1,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(1))
model.add(Activation("sigmoid"))

model.summary()

model.compile(
    optimizer= tf.keras.optimizers.Adam(
        lr=learning_rate,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon,
        decay=weight_decay),
    loss=brLoss(batch_size),
    metrics=[evalLoss, Pearson, Spearman, Kendall]
)

csv_logger = CSVLogger("model_history_log.csv", append=True)
checkpoint = ModelCheckpoint("model.hdf5", monitor='val_loss', verbose=1, save_best_only=False, mode='min')

print("Loading data...")
x_train, y_train, x_test, y_test, x_val, y_val = loadData()
print("Fitting data...")
model.fit(x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_val, y_val),
    callbacks=[csv_logger, checkpoint]
)

prediction = model.predict(x_test)
print(prediction)
print("...................")
print(y_test)

'''
ORIGINAL UNCAHNGED MODEL. DON'T DELETE
model.add(Conv3D(16, input_shape=(120,120,120,11), data_format='channels_last', kernel_size=(3,3,3)))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2)))

model.add(Conv3D(32, kernel_size=(3,3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2)))

model.add(Conv3D(32, kernel_size=(3,3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv3D(64, kernel_size=(3,3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2)))

model.add(Conv3D(128, kernel_size=(3,3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv3D(128, kernel_size=(3,3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv3D(256, kernel_size=(3,3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Conv3D(512, kernel_size=(3,3,3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(3,3,3), strides=(2,2,2)))

model.add(Flatten())

#The input is already 512, thus we reduce it to 256
#This is the 3 fully connected layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(11))
'''