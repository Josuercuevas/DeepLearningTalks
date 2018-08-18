'''
    This file is part of the September 2018 Workshop at Yuan Ze University.

    You can use these examples in the way you seem fit, though I can't make sure
    it will work fine in your case.
'''

import tensorflow as tf
from numpy import array

a = tf.placeholder("float", shape=[3, 3, 3]) # Create a symbolic variable 'a'
# a = tf.placeholder(tf.float32, shape=[3, 3, 3]) # Create a symbolic variable 'a'
# a = tf.placeholder(tf.float16, shape=[3, 3, 3]) # Create a symbolic variable 'a'

a_numpy = array([
  [[1, 2, 3],    [4, 5, 6],    [7, 8, 9]],
  [[11, 12, 13], [14, 15, 16], [17, 18, 19]],
  [[21, 22, 23], [24, 25, 26], [27, 28, 29]],
  ])

# create a session to evaluate the symbolic expressions
with tf.Session() as sess:
    # check tensor information
    print("shape of A inside Tensorflow is: ")
    print(a.get_shape()) # ---> WHY
    print(a.get_shape().as_list()) # ----> WHY
    print(a)

    print("\nThe answer got from evaluating tensor A is:\n")
    # eval expressions with parameters for a
    print(sess.run(a, feed_dict={a: a_numpy}))
