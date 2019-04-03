import keras
from keras.models import Sequential
from keras.layers import Dense, Conv3D, MaxPooling3D, BatchNormalization, Activation, Flatten

from batchRankingLoss import brLoss
from dataGenerator import generateData

batch_size = 1
max_epoch = 50
beta1 = 0.9
beta2 = 0.999
epsilon = 1E-8
learning_rate = 0.0001
weight_decay = 0.0
epochs = 1
#learning_rate_decay = 0.0
#coef_L1 = 0.00001

(x_train, y_train), (x_test, y_test) = generateData()


model = Sequential()

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
model.add(Dense(1))

model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(
        lr=learning_rate,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon,
        decay=weight_decay),
    loss=brLoss(batch_size),
    metrics=["acc"]
)

model.fit(x_train, y_train,
    epochs=epochs,
    batch_size=batch_size,
    validation_data=(x_test, y_test)
)