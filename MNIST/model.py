import tensorflow as tf
import input_data # 调用input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# x不是一个特定的值，而是一个占位符placeholder
# None表示此张量的第一个维度可以是任何长度的,即任意张数的图片
x = tf.placeholder("float", [None, 784])

# [None,784] x [784,10] + [10] = [None,10]

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x,W) + b)

y_real = tf.placeholder("float", [None,10])

cross_entropy = -tf.reduce_sum( y_real * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_real: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_real,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

print("-----------------------")
print("MNIST accuracy:")
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_real: mnist.test.labels}))