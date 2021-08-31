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


# # Attention Block
# def Temporal_attention_block(inputs):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     # print("input",inputs.shape)
#     a = LSTM(nb_input_vector, return_sequences=True)(inputs)
#     a = Permute((2, 1))(a)
#     a = Reshape((nb_input_vector, nb_time_steps))(a) # this line is not useful. It's just to know which dimension is what.
#     a = Dense(nb_time_steps, activation='relu', kernel_regularizer=regularizers.l1(0.01))(a)
#     a_probs = Permute((2, 1), name='tattention_vec')(a)
#     output_attention_mul = Multiply()([inputs, a_probs])
#     return output_attention_mul
#
# def Spatial_attention_block(inputs):
#     # inputs.shape = (batch_size, time_steps, input_dim)
#     # print("input",inputs.shape)
#     a = LSTM(nb_input_vector, return_sequences=True)(inputs)
#     a = Dense(nb_input_vector, activation='softmax', kernel_regularizer=regularizers.l1(0.01))(a)
#     a_probs = Permute((1, 2), name='sattention_vec')(a)
#     # a_probs = Dropout(0.2)(a_probs)
#     output_attention_mul = Multiply()([inputs, a_probs])
#     out = Dense(64, activation='tanh')(output_attention_mul)
#     out = BatchNormalization()(out)
#     return out
#
#
#
# # Build Model
# def BuildModel():
#     K.clear_session()
#     inputs = Input(shape=(nb_time_steps, nb_input_vector,))
#     hidden = BatchNormalization()(inputs)
#     attention_spatial = Spatial_attention_block(hidden)
#     hidden = LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(attention_spatial)
#     hidden = LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(hidden)
#     hidden = LSTM(nb_input_vector, return_sequences=True)(hidden)
#     # hidden = Dropout(0.2)(hidden)
#     attention_time = Temporal_attention_block(hidden)
#     attention_time = Dense(64, kernel_regularizer=regularizers.l1(0.01))(attention_time)
#     # hidden = BatchNormalization()(hidden)
#     flatten = Flatten()(attention_time)
#     output = Dense(60, activation='softmax')(flatten)
#     mod = Model(inputs=[inputs], outputs=output)
#     return mod


inputs = Input(shape=(nb_time_steps, nb_input_vector,))
nor = BatchNormalization()(inputs)

# Spatial_attention
lstm = LSTM(nb_input_vector, return_sequences=True)(nor)
dense = Dense(nb_input_vector, activation='softmax', kernel_regularizer=regularizers.l1(0.01))(lstm)
dense = BatchNormalization()(dense)
a_probs = Permute((1, 2), name='sattention_vec')(dense)
# a_probs = Dropout(0.2)(a_probs)
output_attention_mul = Multiply()([nor, a_probs])
Spatial_attention_out = Dense(64, activation='tanh', kernel_regularizer=regularizers.l1(0.01))(output_attention_mul)
Spatial_attention_out = BatchNormalization()(Spatial_attention_out)

lstm = LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(Spatial_attention_out)
lstm = Dropout(0.2)(lstm)
lstm = LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(lstm)
lstm = Dropout(0.2)(lstm)
lstm = LSTM(64, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(lstm)
lstm = BatchNormalization()(lstm)

# Temporal_attention
lstm_in = LSTM(nb_input_vector, return_sequences=True)(lstm)
a = Permute((2, 1))(lstm_in)
a = Reshape((nb_input_vector, nb_time_steps))(a) # this line is not useful. It's just to know which dimension is what.
dense = Dense(nb_time_steps, activation='relu', kernel_regularizer=regularizers.l1(0.01))(a)
dense = BatchNormalization()(dense)
a_probs = Permute((2, 1), name='tattention_vec')(dense)
output_attention_mul = Multiply()([lstm_in, a_probs])
output_attention_mul = BatchNormalization()(output_attention_mul)

dense = Dense(64, kernel_regularizer=regularizers.l1(0.01))(output_attention_mul)
dense = BatchNormalization()(dense)
flatten = Flatten()(dense)
output = Dense(60, activation='softmax', kernel_regularizer=regularizers.l1(0.01))(flatten)
model = Model(inputs=[inputs], outputs=output)




# model.add(LSTM(units=250, input_shape=(nb_time_steps, nb_input_vector), return_sequences=True))
# model.add(LSTM(units=125, return_sequences=True, kernel_regularizer=regularizers.l2(0.01)))
# model.add(LSTM(units=60,  kernel_regularizer=regularizers.l2(0.01)))
# model.add(Dense(60, activation='softmax'))


# model = BuildModel()
# compile:loss, optimizer, metrics
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# train: epcoch, batch_size
history = model.fit(x_train, y_train, epochs=500, batch_size=32, verbose=1, validation_data=(x_val, y_val))

print(history.history.keys())

print(model.summary())

plot_model(model, to_file='STA-LSTM.png')

score = model.evaluate(x_test, y_test,batch_size=32, verbose=1)
print(score)

# 保存模型
model.save("STA-LSTM2.h5")

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('STA-LSTM_acc2.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig('STA-LSTM_loss2.png')
plt.show()