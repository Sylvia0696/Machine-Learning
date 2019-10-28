import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.layers import Dense,Activation
from keras.models import Sequential


df=pd.read_csv('hw5-trainingset-zr2209.csv', sep=',',header=None)
df_test = pd.read_csv('hw5-testset-zr2209.csv', sep=',',header=None)

data = df.values
data_test = df_test.values


def is_number(s):
    try:  # 如果能运行float(s)语句，返回True（字符串s是浮点数）
        float(s)
        return True
    except ValueError:  # ValueError为Python的一种标准异常，表示"传入无效的参数"
        pass  # 如果引发了ValueError这种异常，不做任何事情（pass：不做任何事情，一般用做占位语句）
    try:
        import unicodedata  # 处理ASCii码的包
        unicodedata.numeric(s)  # 把一个表示数字的字符串转换为浮点数返回的函数
        return True
    except (TypeError, ValueError):
        pass
    return False




for i in range(0, 380):
    answer = set()
    is_num = True
    for j in range(1, 20001):
        if type(data[j][i]) == str:
#             print(type(data[j][i]),j,i)
            if not is_number(data[j][i]):
                is_num = False
                break
    print(is_num, i)
    if is_num == True:
        for j in range(1, 20001):
            if type(data[j][i]) != str:
                continue
            else:
                data[j][i] = float(data[j][i])
                
        for j in range(1, 20001):
            if type(data_test[j][i]) != str or i == 100:
                continue
            else:
                data_test[j][i] = float(data_test[j][i])
    else:
        for j in range(1, 20001):
            answer.add(data[j][i])
        f_to_num = {}
        n = 0
        for ans in answer:
            f_to_num[ans] = n
            n = n + 1
        for j in range(1, 20001):
            if type(data[j][i]) != str:
                continue
            else:
                data[j][i] = f_to_num[data[j][i]]
                
        for j in range(1, 20001):
            if type(data_test[j][i]) != str:
                continue
            elif data_test[j][i] in answer:
                data_test[j][i] = f_to_num[data_test[j][i]]
            else:
                data_test[j][i] = 0





data[pd.isnull(data)] = 0
data_non = data
y = data_non[1:20001,100]
x = np.concatenate((data_non[1:20001,0:100],data_non[1:20001,101:380]),axis = 1)
scaler = StandardScaler() # 标准化转换
scaler.fit(x)  # 训练标准化对象
X = scaler.transform(x)   # 转换数据集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = .1, random_state = 0)



#加激活函数的方法1：mode.add(Activation(''))


#构建一个顺序模型
model=Sequential()

#在模型中添加一个全连接层
#units是输出维度,input_dim是输入维度(shift+两次tab查看函数参数)
model.add(Dense(units=256,input_dim=379))
model.add(Activation('relu'))
model.add(Dense(units=128))
model.add(Activation('relu'))
model.add(Dense(units=64))
model.add(Activation('relu'))
model.add(Dense(units=32))
model.add(Activation('relu'))
model.add(Dense(units=16))
model.add(Activation('relu'))
model.add(Dense(units=1))
# model.add(Activation('linear'))

#定义优化算法(修改学习率)
# defsgd=SGD(lr=0.003)
adam = Adam(lr=0.003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#编译模型
model.compile(optimizer=adam,loss='mse')   #optimizer参数设置优化器,loss设置目标函数

#训练模型
model.fit(X_train, y_train, batch_size = 4, epochs = 30, validation_data = (X_val, y_val), verbose = 1, shuffle=True)



data_test[pd.isnull(data_test)] = 0
x_te = np.concatenate((data_test[1:20001,0:100],data_test[1:20001,101:380]),axis = 1)
scaler.fit(x_te)  # 训练标准化对象
X_te = scaler.transform(x_te)   # 转换数据集
y_tepred = model.predict(X_te)
df_test = pd.read_csv('hw5-testset-zr2209.csv', sep=',',header=None)
data_out = df_test.values


for i in range(0, len(y_tepred)):
    data_out[i+1,100] = y_tepred[i][0]


dff = pd.DataFrame(data_out)
dff.to_csv('hw5-testset-zr2209-submission.csv',index=0,header=0)