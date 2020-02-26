import warnings

warnings.filterwarnings('ignore')

import pandas as pd
import seaborn as sns

def normalize_feature(df):
    return df.apply(lambda column:(column-column.mean()) / column.std())

sns.set(context="notebook",style="whitegrid",palette="dark")
df1 = pd.read_csv("data1.csv",names=["square","bedrooms","price"])
df = normalize_feature(df1)
#print(df.head())
#print(df.info())
# sns.lmplot("square","price",df0,height=6,fit_reg=True)

import numpy as np
ones = pd.DataFrame({"ones":np.ones(len(df))})
df = pd.concat([ones,df],axis=1)
print('===data array===')
print(df.head())

X_data = np.array(df[df.columns[0:3]])
y_data = np.array(df[df.columns[-1]]).reshape(len(df),1)
print('===split into x and y===')
print(X_data.shape,type(X_data))
print(y_data.shape,type(y_data))

import tensorflow as tf
alpha = 0.01 # 学习率 alpha
epoch = 500 # 训练全量数据集的轮数

with tf.name_scope('input'):
    X = tf.compat.v1.placeholder(tf.float32, X_data.shape, name='X')
    y = tf.compat.v1.placeholder(tf.float32, y_data.shape, name='y')

with tf.name_scope('hypothesis'):
    # 权重变量 W，形状[3,1]
    W = tf.compat.v1.get_variable("weights", (3, 1), initializer=tf.constant_initializer())
    # 假设函数 h(x) = w0*x0+w1*x1+w2*x2, 其中x0恒为1
    # 推理值 y_pred  形状[47,1]
    y_pred = tf.matmul(X, W, name='y_pred')

with tf.name_scope('loss'):
    # 损失函数采用最小二乘法，y_pred - y 是形如[47, 1]的向量。
    # tf.matmul(a,b,transpose_a=True) 表示：矩阵a的转置乘矩阵b，即 [1,47] X [47,1]
    # 损失函数操作 loss
    loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y_pred - y), transpose_a=True)

with tf.name_scope('train'):
    # 随机梯度下降优化器 opt
    train_op = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss_op)

with tf.compat.v1.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    # 创建FileWriter实例，并传入当前会话加载的数据流图
    writer = tf.summary.FileWriter('./summary/linear-regression-1', sess.graph)
    # 记录所有损失值
    loss_data = []
    # 开始训练模型
    # 因为训练集较小，所以采用批梯度下降优化算法，每次都使用全量数据训练
    for e in range(1, epoch + 1):
        _, loss, w = sess.run([train_op, loss_op, W], feed_dict={X: X_data, y: y_data})
        # 记录每一轮损失值变化情况
        loss_data.append(float(loss))
        if e % 10 == 0:
            log_str = "Epoch %d \t Loss=%.4g \t Model: y = %.4gx1 + %.4gx2 + %.4g"
            print(log_str % (e, loss, w[1], w[2], w[0]))

# 关闭FileWriter的输出流
writer.close()  





