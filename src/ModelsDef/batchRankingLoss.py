from keras import backend as K

def brLoss(batch):
    
    def greaterThanApprox(x, y):
        epsilon = 1E-8
        abse = np.sqrt((x - y)**2 + epsilon)
        return 0.5 * (x + y + abse)

    def piecewiseFunction1(x, y):
        epsilon = 1E-8
        threshold = 1E-1
        approx = greaterThanApprox(x, y)

        check = approx - (x * (1 + threshold))

        #For checking if x == y
        same = x - y
        sameFactor = (same / (same + epsilon))

        value = np.abs(check) / check 
        value -= (2 * (sameFactor - 1))

        return value

    def piecewiseFunction2(x, y):
        epsilon = 1E-8
        threshold = 1E-1
        approx = greaterThanApprox(x, y)

        check = approx - (x * (1 + threshold))

        #For checking if x == y
        same = x - y
        sameFactor = same / (same + epsilon)

        value = (check - np.abs(check)) / ((check - np.abs(check)) + epsilon)
        value *= sameFactor

        return value
    
    def batchRankingLoss(y_true, y_pred):
        """ Final loss calculation function to be passed to optimizer"""
        L = y_pred[0,0] * 0.0
        #y_true = K.tf.reshape(y_true, [-1])
        #y_pred = K.tf.reshape(y_pred, [-1])

        for i in range(batch):
            for j in range(batch):
                if i != j:
                    
                    #y_ij = K.tf.cond(K.tf.greater(y_true[i], y_true[j]), lambda: K.tf.constant(-1.0), lambda: K.tf.constant(1.0))
                    y_ij = piecewiseFunction1(y_true[i,0], y_true[j,0])

                    #w_ij = K.tf.cond(K.tf.greater(K.tf.abs(y_true[i] - y_true[j]), K.tf.constant(0.1)), lambda: K.tf.constant(1.0), lambda: K.tf.constant(0.0))
                    w_ij = piecewiseFunction2(K.tf.abs(y_true[i,0] - y_true[j,0]), K.tf.constant(0.1))

                    L_ij = w_ij * (K.tf.reduce_max(K.tf.stack([0.0, (1.0 - y_ij) * (y_pred[i,0] - y_pred[j,0])])))
                    #print(L_ij.shape)
                    L += L_ij

        return K.tf.multiply(1.0/(batch*batch), K.tf.reduce_sum(L))

    return batchRankingLoss