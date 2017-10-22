import tensorflow as tf

with tf.Session() as sess:
    print(sess.run(tf.zeros([10])))
