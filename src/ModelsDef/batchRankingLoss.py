import tensorflow as tf

def brLoss(batch, score_threshold = 0.1):
    
    def greaterThanApprox(x, y):
        epsilon = 1E-8
        abse = tf.sqrt((x - y)**2 + epsilon)
        return 0.5 * (x + y + abse)

    def piecewiseFunction1(x, y):
        epsilon = 1E-8
        threshold = 1E-1
        approx = greaterThanApprox(x, y)

        check = approx - (x * (1 + threshold))

        #For checking if x == y
        same = x - y
        sameFactor = (same / (same + epsilon))

        value = tf.abs(check) / check 
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

        value = (check - tf.abs(check)) / ((check - tf.abs(check)) + epsilon)
        value *= sameFactor

        return value
    
    def batchRankingLoss(y_true, y_pred):
        """ Final loss calculation function to be passed to optimizer"""
        L = y_pred[0] * 0.0
        #y_true = K.tf.reshape(y_true, [-1])
        #y_pred = K.tf.reshape(y_pred, [-1])

        # #Gets us the batch size of current run
        # temp = K.tf.Variable(K.tf.zeros(batch, dtype=K.tf.float32))
        # batch_size = temp[:K.int_shape(y_true)[0]]
        
        # i = K.tf.constant(0)
        # def cond(i, ii):
        #     return K.tf.less(i, ii)

        # #L = K.tf.while_loop(cond, body, [batch_size, L])

        for i in range(batch):
            for j in range(batch):
                if i != j:
                    
                    #y_ij = K.tf.cond(K.tf.greater(y_true[i], y_true[j]), lambda: K.tf.constant(-1.0), lambda: K.tf.constant(1.0))
                    y_ij = piecewiseFunction1(y_true[i], y_true[j])

                    #w_ij = K.tf.cond(K.tf.greater(K.tf.abs(y_true[i] - y_true[j]), K.tf.constant(0.1)), lambda: K.tf.constant(1.0), lambda: K.tf.constant(0.0))
                    w_ij = piecewiseFunction2(tf.abs(y_true[i] - y_true[j]), tf.constant(score_threshold))

                    L_ij = w_ij * (tf.reduce_max(tf.concat([[0.0], (1.0 - y_ij) * (y_pred[i] - y_pred[j])], axis=0)))
                    #print(L_ij.shape)
                    L += L_ij

        return tf.multiply(1.0/(batch*batch), tf.reduce_sum(L))

    return batchRankingLoss