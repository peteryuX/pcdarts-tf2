import tensorflow as tf


def CrossEntropyLoss():
    """"cross entropy loss"""
    def cross_entropy_loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                            logits=y_pred)

        return tf.reduce_mean(ce)
    return cross_entropy_loss
