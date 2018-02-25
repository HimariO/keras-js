import numpy as np
import tensorflow as tf

with tf.Graph().as_default():
    with tf.Session() as sess:
        conv_out = tf.placeholder(tf.int32, shape=[1, 38, 38, 8])
        reshaped = tf.space_to_depth(conv_out, block_size=2)

        indexMap = np.zeros([1, 38, 38, 8], dtype=np.int32)
        for i in range(8):
            indexMap[0, :, :, i] = np.arange(np.prod([38, 38]), dtype=np.int32).reshape([38, 38]) + i * 38**2
        print(indexMap[0, :, :, 0])
        print(indexMap[0, :, :, 3])

        result = sess.run(reshaped, feed_dict={
            conv_out: indexMap
        })

        # for i in range(2):
        print(result[:, 0, 0, :])
        print(result.shape)
