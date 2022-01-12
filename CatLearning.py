import matplotlib.pyplot as plt
import numpy as np
import os

from lr_utils import load_dataset
from model.Model import predict, optimize, initialize_with_zeros

trainSet_x_orig, trainSet_y, testSet_x_orig, testSet_y, classes = load_dataset()

m_train = trainSet_y.shape[1]  # 训练集里图片的数量。
m_test = testSet_y.shape[1]  # 测试集里图片的数量。
num_px = trainSet_x_orig.shape[1]  # 训练、测试集里面的图片的宽度和高度（均为64x64）。

print("训练集的数量: m_train = " + str(m_train))
print("测试集的数量 : m_test = " + str(m_test))
print("每张图片的宽/高 : num_px = " + str(num_px))
print("每张图片的大小 : (" + str(num_px) + ", " + str(num_px) + ", 3)")
print("训练集_图片的维数 : " + str(trainSet_x_orig.shape))
print("训练集_标签的维数 : " + str(trainSet_y.shape))
print("测试集_图片的维数: " + str(testSet_x_orig.shape))
print("测试集_标签的维数: " + str(testSet_y.shape))

# X_flatten = X.reshape(X.shape [0]，-1).T ＃X.T是X的转置
# 将训练集的维度降低并转置。
trainSet_x_flatten = trainSet_x_orig.reshape(trainSet_x_orig.shape[0], -1).T
# 将测试集的维度降低并转置。
testSet_x_flatten = testSet_x_orig.reshape(testSet_x_orig.shape[0], -1).T

print("训练集降维最后的维度 : " + str(trainSet_x_flatten.shape))
print("训练集_标签的维数 : " + str(trainSet_y.shape))
print("测试集降维之后的维度: " + str(testSet_x_flatten.shape))
print("测试集_标签的维数 : " + str(testSet_y.shape))

# 标准化数据集
trainSet_x = trainSet_x_flatten / 255
testSet_x = testSet_x_flatten / 255

# using a single func model to simplify the call of functions
def train(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False, save_by_steps = False):
    """
    通过调用之前实现的函数来构建逻辑回归模型
    
    参数：
        X_train  - numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
        Y_train  - numpy的数组,维度为（1，m_train）（矢量）的训练标签集
        X_test   - numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
        Y_test   - numpy的数组,维度为（1，m_test）的（向量）的测试标签集
        num_iterations  - 表示用于优化参数的迭代次数的超参数
        learning_rate  - 表示optimize（）更新规则中使用的学习速率的超参数
        print_cost  - 设置为true以每100次迭代打印成本
    
    返回：
        d  - 包含有关模型信息的字典。
    """
    w, b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost, save_by_steps)

    # 从字典“参数”中检索参数w和b
    w, b = parameters["w"], parameters["b"]

    # 预测测试/训练集的例子
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性
    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}
    return d

if __name__ == "__main__":
    if not os.path.exists('./output'):
        os.mkdir('./output')
    print("====================测试model====================")
    d = train(trainSet_x, trainSet_y, testSet_x, testSet_y, num_iterations=2000, learning_rate=0.005, print_cost=True, save_by_steps=False)

    # 保存所得参数
    np.savez('output/params.npz', d['w'], d['b'])
    print("Model saved at output/params.npz")

    # 绘制图
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(d["learning_rate"]))
    plt.show()
