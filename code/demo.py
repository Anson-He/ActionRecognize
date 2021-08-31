from keras.models import load_model
import numpy as np
import poseModule as pm
import cv2
import pandas as pd

cap = cv2.VideoCapture('jump.mp4')
model = load_model('STA-LSTM2.h5')
a = np.random.rand(100, 57)
a = a.reshape(1, 100, 57)
model.predict(a)
detector = pm.poseDetector()
need = [0, 12, 11, 14, 13, 16, 15, 20, 19, 18, 17, 24, 23, 26, 25, 28, 27, 32, 31]
action_names = ['drink water', 'eat meal', 'brushing teeth', 'brushing hair', 'drop', 'pickup', 'throw', 'giving project', 'standing up', 'clapping', 'walking towards each other', 'writing', 'tear up paper', 'hand waving', 'take off jacket', 'wear a shoe', 'take off a shoe', 'wear on glasses', 'standing up', 'put on a hat', 'take off a hat', 'cheer up', 'hand waving', 'kicking something', 'jump up', 'hopping', 'reach into pocket', 'make a phone call', 'playing with phone', 'typing on a keyboard', 'pointing to something with finger', 'taking a selfie', 'check time from watch', 'rub two hands together', 'nod head', 'shake head', 'wipe face', 'salute', 'put the palms together', 'cross hands in front', 'sneeze', 'staggering', 'falling', 'touch head (headache)', 'touch chest (heart pain)', 'touch back (backache)', 'touch neck (neckache)', 'nausea', 'use a fan', 'punching other person', 'kicking other person', 'pushing other person', 'pat on back of other person', 'point finger at the other person', 'hugging other person', 'giving something to other person', 'touch some person pocket', 'handshaking', 'sitting down', 'walking towards each other' ]

frame = []
video = []
while True:
    success, img = cap.read()
    img = detector.findPose(img)
    lmList = detector.findPostion(img)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
    for i in range(len(lmList)):
        if i in need:
            for j in lmList[i]:
                frame.append(j)
    video.append(frame)
    frame = []
    if len(video) == 100:
        df = pd.DataFrame(video)
        df = (df - df.min()) / (df.max() - df.min())
        video = np.array(df)
        video = video.reshape(1, 100, 57)
        print(video)
        pred = model.predict(video)
        video = []
        pred2 = np.argmax(pred, axis=1)
        print(pred2[0])
        print(action_names[pred2[0]])
        print(pred)



# data = np.load('finaldata.npy')
# label = np.load('totallabel.npy')
# print(data.shape)
# x = data[1]
# y = label[1]
# x = x.reshape(1,100,57)
# # model = load_model('STA-LSTM2.h5')
# preds = model.predict(x)
# print(list(preds))
# print(y)