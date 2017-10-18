# Python
import tensorflow as tf

hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
string = sess.run(hello).decode('utf-8')
print(string)
