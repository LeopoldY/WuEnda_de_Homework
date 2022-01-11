import numpy as np

# 构建sigmoid函数
def sigmoid(z): 
    """
    参数：
        z  - 任何大小的标量或numpy数组。
    
    返回：
        s  -  sigmoid（z）
    """
    y = 1 / (1 + np.exp(-z))
    return y

# using w & b to predict the tag Y in the data set
def predict(w, b, X):
    """
    使用学习逻辑回归参数logistic （w，b）预测标签是0还是1，
    
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据
    
    返回：
        Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）
    
    """

    m = X.shape[1]  # 图片的数量
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # 计预测猫在图片中出现的概率
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # 将概率a [0，i]转换为实际预测p [0，i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
    # 使用断言
    assert (Y_prediction.shape == (1, m))

    return Y_prediction

def propagate(w, b, X, Y):
    """
    实现前向和后向传播的成本函数及其梯度。
    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 矩阵类型为（num_px * num_px * 3，训练数量）
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据数量)

    返回：
        cost- 逻辑回归的负对数似然成本
        dw  - 相对于w的损失梯度，因此与w的形状相同
        db  - 相对于b的损失梯度，因此与b的形状相同
    """
    m = X.shape[1]

    # 正向传播
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))  # 计算成本

    # 反向传播
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {
        "dw": dw,
        "db": db
    }

    return grads, cost

# 用梯度下降法来优化w&b
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False, save_by_steps = False):
    """
        此函数通过运行梯度下降算法来优化w和b
        
        参数：
            w  - 权重，大小不等的数组（num_px * num_px * 3，1）
            b  - 偏差，一个标量
            X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
            Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
            num_iterations  - 优化循环的迭代次数
            learning_rate  - 梯度下降更新规则的学习率
            print_cost  - 每100步打印一次损失值
        
        返回：
            params  - 包含权重w和偏差b的字典
            grads  - 包含权重和偏差相对于成本函数的梯度的字典
            成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。
        
        提示：
        我们需要写下两个步骤并遍历它们：
            1）计算当前参数的成本和梯度，使用propagate（）。
            2）使用w和b的梯度下降法则更新参数。
    """

    costs = []

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            costs.append(cost)
            if save_by_steps:
                np.savez(f'output/params_{i}.npz', w, b) 
        # 打印成本数据
        if print_cost and (i % 100 == 0):
            print("迭代的次数: %i ， 误差值： %f" % (i, cost))

    params = {
        "w": w,
        "b": b
    }

    grads = {
        "dw": dw,
        "db": db
    }

    return params, grads, costs

    # initialize parm w & b
def initialize_with_zeros(dim):
    """
        此函数为w创建一个维度为（dim，1）的0向量，并将b初始化为0。
        
        参数：
            dim  - 我们想要的w矢量的大小（或者这种情况下的参数数量）
        
        返回：
            w  - 维度为（dim，1）的初始化向量。
            b  - 初始化的标量（对应于偏差）
    """
    w = np.zeros(shape=(dim, 1))
    b = 0

    # by using "assert" to make sure that my data is correct
    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))  # ensure the type of b is float or int

    return w, b
