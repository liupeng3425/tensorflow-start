import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 28 * 28])
W = tf.Variable(tf.zeros([28 * 28, 10]))
b = tf.Variable(tf.zeros([10]))

# 预测值
y_predict = tf.nn.softmax(tf.matmul(x, W) + b)
# 真实值
y_label = tf.placeholder("float", [None, 10])
# 交叉墒，优化目标
cross_entropy = -tf.reduce_sum(y_label * tf.log(y_predict))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 设置初始化方式
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)  # 执行初始化

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, {x: batch_xs, y_label: batch_ys})

# tf.argmax()它能给出某个tensor对象在某一维上的其数据最大值所在的索引值, axis：0表示按列，1表示按行
correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_label, 1))
# tf.cast() 把boolean转成0和1
# tf.reduce_mean() 求平均数
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, {x: mnist.test.images, y_label: mnist.test.labels}))
sess.close()
