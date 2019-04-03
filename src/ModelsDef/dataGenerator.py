from keras.preprocessing import sequence
import numpy as np

def generateData():

    decoys = 1
    width = 120
    height = 120
    depth = 120
    atom_types = 11

    
    xHigh = np.random.random((decoys,width,height,depth,atom_types)) * 10000
    xLow = np.random.random((decoys,width,height,depth,atom_types))
    x_train = np.append(xHigh, xLow, axis=0)


    y = np.random.uniform(0.5, 1.0, decoys)
    y2 = np.random.uniform(0.0, 0.2, decoys)
    y_train = np.append(y, y2, axis=0)

    randomize = np.arange(len(y_train))
    np.random.shuffle(randomize)
    x_train = x_train[randomize]
    y_train = y_train[randomize]


    XHigh = np.random.random((decoys,width,height,depth,atom_types)) * 10000
    xLow = np.random.random((decoys,width,height,depth,atom_types))
    x_test = np.append(xHigh, xLow, axis=0)

    y = np.random.uniform(0.5, 1.0, decoys)
    y2 = np.random.uniform(0.0, 0.2, decoys)
    y_test = np.append(y, y2, axis=0)

    randomize = np.arange(len(y_test))
    np.random.shuffle(randomize)
    x_test = x_test[randomize]
    y_test = y_test[randomize]

    #Here we return 2 tuples with our generated data
    return (x_train, y_train), (x_test, y_test)
