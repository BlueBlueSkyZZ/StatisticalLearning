import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
'''
inputs:输入
in_size:输入数据大小（神经元个数）
out_size:输出数据大小（神经元个数）
activate_function:激励函数
'''
def add_layer(inputs, in_size, out_size, n_layer,activate_function=None):
    with tf.name_scope('layer'):
        layer_name = 'layer%s' % n_layer
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name+'/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activate_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activate_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs

x_data = np.linspace(-1, 1, 300)[:,np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# hidden layer
l1 = add_layer(xs, 1, 10, n_layer=1, activate_function=tf.nn.relu)
# output layer
predition = add_layer(l1, 10, 1, n_layer=2, activate_function=None)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),
                                    reduction_indices=[1]))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(0.2)
    optimizer2 = tf.train.AdamOptimizer(0.2)
    train_step = optimizer.minimize(loss)

init = tf.global_variables_initializer()

fig = plt.figure() # 生成一个图片框
ax = fig.add_subplot(1,1,1) # 编号
ax.scatter(x_data, y_data) # 输入真实值(点的形式)
plt.ion() # 互动模式开启show后不暂停
plt.show()

merge = tf.summary.merge_all() # 合并所有的tensorborad操作

with tf.Session() as sess:
    sess.run(init)

    writer = tf.summary.FileWriter("logs/", sess.graph)

    for i in range(1000):
        sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))
            try:
                ax.lines.remove(lines[0])  #抹除
            except Exception:
                pass
            predition_value = sess.run(predition, feed_dict={xs:x_data})
            lines = ax.plot(x_data, predition_value, 'r-', lw=5) # 线的形式

            plt.show()
            plt.pause(0.1)  # 暂停0.1s

            result = sess.run(merge, feed_dict={xs:x_data, ys:y_data})
            writer.add_summary(result, i)
        if i == 999:
            plt.pause(5)