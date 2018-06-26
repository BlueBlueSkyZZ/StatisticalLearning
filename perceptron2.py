import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection

'''
利用高斯白噪声生成基于某个直线附近的若干个点
y = wb + b
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

def trainByStochasticGradientDescent(input, output, input_num, train_num = 10000, learning_rate = 1):
    global Weight, Bias
    x = input
    y = output
    for rounds in range(train_num):
        for i in range(input_num):
            x1, x2 = x[i]
            prediction = np.sign(Weight[0] * x1 + Weight[1] * x2 + Bias)
            # print("prediction", prediction)
            if y[i] * prediction <= 0: # 判断误分类点
                # Weight = Weight + np.reshape(learning_rate * y[i] * x[i], (2,1))
                Weight[0] = Weight[0] + learning_rate * y[i] * x1
                Weight[1] = Weight[1] + learning_rate * y[i] * x2
                Bias = Bias + learning_rate * y[i]
                # print(Weight, Bias)
                draw_line(Weight, Bias)
                break
        if rounds % 200 == 0:
            learning_rate *= 0.5
            if compute_accuracy(input, output, input_num, Weight, Bias) == 1:
                print("rounds:", rounds)
                break;


def draw_line(Weight, Bias):
    global ax, lines
    x1, y1 = -1, ((-Bias-Weight[0]*(-1) ) / Weight[1])[0]
    x2, y2 = 1, ((-Bias-Weight[0]*(1) ) / Weight[1])[0]
    try:
        ax.lines.remove(lines[0])  # 抹除
    except Exception:
        # plt.pause(0.1)
        pass
    lines = ax.plot([x1, x2], [y1, y2], 'r-', lw=5)  # 线的形式
    # lines = ax.plot([-1, 1], [2, 4 + 5*np.random.rand(1)], 'r-', lw=5)  # 线的形式
    plt.show()
    plt.pause(0.01)

def compute_accuracy(x_test, y_test, test_size, weight, bias):
    x1, x2 =  np.reshape(x_test[:,0], (test_size, 1)), np.reshape(x_test[:,1], (test_size, 1))
    prediction = np.sign(y_test * ( x1 * weight[0] + x2 * weight[1] + bias ))
    count = 0
    # print("prediction",prediction)
    for i in range(prediction.size):
        if prediction[i] > 0:
            count = count + 1
    return (count+0.0)/test_size

# 直线的真正参数
real_weight = 1
real_bias = 3
size = 100
testSize = 15

# 初始化w、b
Weight = np.random.rand(2,1) # 随机生成-1到1一个数据
Bias = 0 # 初始化为0

# 输入数据和标签
# 生成输入的数据
input_point = random_point_nearby_line(real_weight, real_bias, size)
# 给数据打标签，在直线之上还是直线之下，above=1,below=-1
label = np.sign(input_point[:,1] - (input_point[:,0] * real_weight + real_bias)).reshape((size, 1))

x_train, x_test, y_train, y_test = model_selection.train_test_split(input_point, label, test_size=testSize)
trainSize = size - testSize

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
lines = ax.plot([-1 , 1], [-real_weight+real_bias, real_weight+real_bias], 'r-', lw=1)
plt.pause(1.5)

trainByStochasticGradientDescent(x_train, y_train, trainSize)
print("accuracy:", compute_accuracy(x_test, y_test, testSize, Weight, Bias))
print("Weight:",Weight)
print("Bias:",Bias)