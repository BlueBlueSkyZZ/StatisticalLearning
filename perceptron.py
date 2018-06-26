import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

'''
利用高斯白噪声生成基于某个直线附近的若干个点
weight 直线权值
bias 直线偏置
size 点的个数
'''
def random_point_nearby_line(weight , bias , size = 10):
    x_point = np.linspace(-1, 1, size)[:,np.newaxis]
    noise = np.random.normal(0, 0.5, x_point.shape)
    y_point = weight * x_point + bias + noise
    input_arr = np.hstack((x_point, y_point))
    return input_arr


# 直线的真正参数
real_weight = 1
real_bias = 3
size = 30
testSize = 0.5

# 感知机的输入输出
x_input = tf.placeholder(tf.float32, shape=[size, 2])
y_input = tf.placeholder(tf.float32, shape=[size, 1])

# 初始化w、b
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0, dtype=tf.float32)) # 随机生成-1到1一个数据
Biases = tf.Variable(tf.zeros([1], dtype=tf.float32)) # 初始化为0

# 输入数据和标签
# 生成输入的数据
input_point = random_point_nearby_line(real_weight, real_bias, size)
# 给数据打标签，在直线之上还是直线之下，above=1,below=-1
label = np.where(input_point[:,1] > (input_point[:,0] * real_weight + real_bias), 1, -1).reshape((size, 1))

x_train, x_test, y_train, y_test = model_selection.train_test_split(input_point, label, test_size=testSize)
# print(y_train)
# print("x_train:", x_train)
# print("x_test:", x_test)
# print("y_train:", y_train)
# print("y_test:", y_test)

# 将输入点绘图
fig = plt.figure() # 生成一个图片框
ax = fig.add_subplot(1,1,1) # 编号
for i in range(y_train.size):
    if y_train[i] == 1:
        ax.scatter(x_train[i,0], x_train[i,1], color='r') # 输入真实值(点的形式) 红色在线上方
    else:
        ax.scatter(x_train[i, 0], x_train[i, 1], color='b')  # 输入真实值(点的形式) 蓝色在线下方
plt.ion() # 互动模式开启show后不暂停
plt.show()
# initial line
line = ax.plot([-1 , 1], [2 , 4], 'r-', lw=1)

# plt.pause(100)


# prediction = tf.where(tf.greater(x_input[:,1], x_input[:,0] * Weights + Biases), tf.constant(1), -1)
# prediction = tf.reshape(tf.where(label == label, tf.constant(1, dtype=tf.float32), tf.constant(-1, dtype=tf.float32)), [size,1])
print(x_train[:,1], '\n',x_train[:,0]* real_weight + real_bias)
prediction = tf.where(x_train[:,1] > (x_train[:,0] * real_weight + real_bias), tf.constant(1), tf.constant(-1))

# 定义损失函数
judge = tf.not_equal(y_input, prediction)
loss = tf.reduce_sum(tf.where(judge, - y_input * (Weights * x_input[:,0] + Biases), 0))

# 优化器
optimizer = tf.train.ProximalGradientDescentOptimizer(0.05)
train_step = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)

    for i in range(2000):
        sess.run(train_step, feed_dict={x_input:x_train, y_input:y_train})
        if i % 500 == 0:
            # print(i, sess.run(Weights), sess.run(Biases), sess.run(loss,feed_dict={x_input:input_point, y_output:label}))
            print(sess.run(loss,feed_dict={x_input:x_train, y_input:y_train}))
            try:
                ax.lines.remove(lines[0])  #抹除
            except Exception:
                plt.pause(10)
                pass
            prediction_value = sess.run(prediction)
            print(prediction_value)
            lines = ax.plot(input_point, prediction_value, 'b-', lw=5)  # 线的形式
            plt.show()
            plt.pause(0.1)