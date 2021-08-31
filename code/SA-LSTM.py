import tensorflow as tf
from keras.layers import Dense, LSTM, Dropout, Flatten, Normalization, Activation, Permute, Multiply, Reshape, Lambda, \
    RepeatVector, BatchNormalization
import keras.backend as K
from keras import regularizers, Input, Model
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# data = pd.read_csv('newdata.csv')
# del data['Unnamed: 0']
# dd = []
# for i in range(len(data['data'])):
#     print(i)
#     a = eval(data['data'][i])
#     dd.append(a)
#
# data['data'] = dd
# x = data['data']
# y = data['label']

# # 标签向量化
# yy = []
# for i in y:
#     if i == 'Full squat':
#         yy.append(0)
#     if i == 'Quarter squat':
#         yy.append(1)
#     if i == 'Standard squat':
#         yy.append(2)

# yy = np.array(y)
# x = np.array(list(x))


# x_train, x_test, y_train,y_test = train_test_split(x, yy)
# print("-----------------------------")
# print(len(x_train))
# print(len(x_test))
# print("-----------------------------")

# x_train = np.load('skeletons_array_train_S.npy')
# y_train = np.load('skeletons_array_train_labels_S.npy')
# # 划分验证集
# x_train, x_val, y_train,y_val = train_test_split(x_train, y_train, test_size=0.25)
# x_test = np.load('skeletons_array_test_S.npy')
# y_test = np.load('skeletons_array_test_labels_S.npy')

# 读数据
X = np.load('finaldata.npy')
Y = np.load('totallabel.npy')
xx, x_test, yy, y_test = train_test_split(X, Y, test_size=0.2)
x_train, x_val, y_train, y_val = train_test_split(xx, yy, test_size=0.25)


# 标签独热编码
y_train = to_categorical(y_train, num_classes=60)
y_val = to_categorical(y_val, num_classes=60)
y_test = to_categorical(y_test, num_classes=60)


# parameters for LSTM
nb_lstm_outputs = 64  # 神经元个数
nb_time_steps = 100  # 时间序列长度
nb_input_vector = 57 # 输入序列


# Attention Block
def Spatial_attention_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    # print("input",inputs.shape)
    a = LSTM(nb_input_vector, return_sequences=True)(inputs)
    a = Dense(nb_input_vector, activation='softmax', kernel_regularizer=regularizers.l1(0.01))(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)
    # a_probs = Dropout(0.2)(a_probs)
    output_attention_mul = Multiply()([inputs, a_probs])
    out = Dense(64, activation='tanh')(output_attention_mul)
    out = BatchNormalization()(out)
    return out


# Build Model
def BuildModel():
    K.clear_session()
    inputs = Input(shape=(nb_time_steps, nb_input_vector,))
    hidden = BatchNormalization()(inputs)
    attention_spatial = Spatial_attention_block(hidden)
    hidden = LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(attention_spatial)
    hidden = LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(hidden)
    hidden = LSTM(64, return_sequences=True)(hidden)
    hidden = Dense(64, kernel_regularizer=regularizers.l1(0.01))(hidden)
    # hidden = BatchNormalization()(hidden)
    flatten = Flatten()(hidden)
    output = Dense(60, activation='softmax')(flatten)
    mod = Model(inputs=[inputs], outputs=output)
    return mod

# model.add(LSTM(units=250, input_shape=(nb_time_steps, nb_input_vector), return_sequences=True))
# model.add(LSTM(units=125, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
# model.add(LSTM(units=60,  kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dense(60, activation='softmax'))


model = BuildModel()
# compile:loss, optimizer, metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# train: epcoch, batch_size
history = model.fit(x_train, y_train, epochs=50, batch_size=64, verbose=1, validation_data=(x_val, y_val))

print(history.history.keys())

print(model.summary())

plot_model(model, to_file='SA-LSTM.png')

score = model.evaluate(x_test, y_test,batch_size=64, verbose=1)
print(score)

# 保存模型
model.save("SA-LSTM.h5")

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('SA-LSTM_acc.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('SA-LSTM_loss.png')
plt.show()