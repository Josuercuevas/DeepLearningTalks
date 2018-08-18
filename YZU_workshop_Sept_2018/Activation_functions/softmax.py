'''
    This file is part of the September 2018 Workshop at Yuan Ze University.

    You can use these examples in the way you seem fit, though I can't make sure
    it will work fine in your case.
'''

import numpy as np
import tensorflow as tf

input_tensor = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]])

# In Numpy
exp_tensor = np.exp(input_tensor)
sum_expo_tensor = np.sum(exp_tensor)
softmax_tensor = exp_tensor / sum_expo_tensor

print("\nSoftmax result on input tensor is (NUMPY):\n")
print(softmax_tensor)


# In Tensorflow
input_tensor_tf = tf.placeholder("float", shape=[1, 10])
softmax_tensor = tf.contrib.layers.softmax(input_tensor_tf)

with tf.Session() as sess:
    print("\nSoftmax result on input tensor is (TENSORFLOW):\n")
    print(sess.run(softmax_tensor, feed_dict={input_tensor_tf: input_tensor}))
    # print(softmax_tensor)
