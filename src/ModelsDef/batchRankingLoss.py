import tensorflow as tf

def brLoss(batch):    

    def batchRankingLoss(y_true, y_pred):
        """ Final loss calculation function to be passed to optimizer"""
        L = 0.0
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        for i in range(batch):
            for j in range(batch):
                if i != j:
                    y_ij = 0

                    y_ij = tf.cond(tf.greater(y_true[i], y_true[j]), lambda: tf.constant(-1.0), lambda: tf.constant(1.0))

                    w_ij = tf.cond(tf.greater(tf.abs(y_true[i] - y_true[j]), tf.constant(0.1)), lambda: tf.constant(1.0), lambda: tf.constant(0.0))

                    L_ij = w_ij * (tf.reduce_max(tf.stack([tf.constant(0.0), (1.0 - y_ij) * (y_pred[i] - y_pred[j])])))

                    L += L_ij

        return (1/(batch**2)) * L

    return batchRankingLoss