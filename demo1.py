import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

### tf structure

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0)) # 随机生成-1到1一维数据
biases = tf.Variable(tf.zeros([1])) # 初始化为0

y = Weights * x_data + biases
loss = tf.reduce_mean(tf.square(y - y_data)) # 定义损失函数
optimizer = tf.train.GradientDescentOptimizer(0.5) # 梯度下降优化器
train = optimizer.minimize(loss) # 最小化误差

init = tf.global_variables_initializer() # 初始化变量

### tf structure

## create Session
sess = tf.Session()
sess.run(init) # 激活
print(sess.run(Weights), sess.run(biases))
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases), sess.run(loss))