import numpy as np
import pandas as pd
label2 = np.load('skeletons_array_test_labels_S.npy')
xx = np.load('skeletons_array_train_S.npy')
label = np.load('skeletons_array_train_labels_S.npy')
xx2 = np.load('skeletons_array_test_S.npy')
xx = np.array(list(xx)+list(xx2))
labels = np.array(list(label)+list(label2))
newxx = []
need = [3, 8, 4, 9, 5, 10, 6, 24, 22, 23, 21, 16, 12, 17, 13, 18, 14, 19, 15]
for i in range(len(xx)):
    samples = []
    for j in range(len(xx[i])):
        frame = []
        count = 0
        old = 0
        flag = True
        for k in range(len(xx[i][j])):
            if k-old == 3:
                old = k
                count = count + 1
                flag = True
            if count in need and flag == True:
                frame.append(xx[i][j][k])
                frame.append(xx[i][j][k+1])
                frame.append(xx[i][j][k+2])
                flag = False
        samples.append(frame)
    newxx.append(samples)
# 归一化
ll = []
for a in newxx:
    df = pd.DataFrame(a)
    df = (df - df.min()) / (df.max() - df.min())
    l = np.array(df).tolist()
    ll.append(l)
print(len(ll))
print(len(ll[0]))
print(len(ll[0][0]))
c = {
    "data": ll,
    "label": labels
}
df = pd.DataFrame(c)
df.to_csv("finaldata.csv")