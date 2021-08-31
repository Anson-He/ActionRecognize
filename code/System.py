from PyQt5 import QtCore, QtGui, QtWidgets
import sys
import cv2
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QFileDialog, QApplication

import poseModule as pm
import pandas as pd
import numpy as np
from keras.models import load_model
import time

class Ui_MainWindow(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)  # 父类的构造函数

        self.timer_camera = QtCore.QTimer()  # 定义定时器，用于控制显示视频的帧率
        self.cap = cv2.VideoCapture()  # 视频流
        self.CAM_NUM = 0  # 为0时表示视频流来自笔记本内置摄像头

        self.set_ui()  # 初始化程序界面
        self.slot_init()  # 初始化槽函数
        self.detector = pm.poseDetector() # 初始化检测器
        self.model = load_model('STA-LSTM2.h5')# 导入模型
        self.pTime = 0 # 计算FPS
        # 预先加载一次，以免第一次识别卡顿
        a = np.random.rand(100, 57)
        a = a.reshape(1, 100, 57)
        self.model.predict(a)

        self.need = [0, 12, 11, 14, 13, 16, 15, 20, 19, 18, 17, 24, 23, 26, 25, 28, 27, 32, 31]
        self.action_names = ['null', 'eat meal', 'brushing teeth', 'brushing hair', 'drop', 'pickup', 'throw',
                        'sitting down', 'standing up', 'clapping', 'walking towards each other', 'writing',
                        'tear up paper', 'hand waving', 'take off jacket', 'wear a shoe', 'take off a shoe',
                        'wear on glasses', 'standing up', 'put on a hat', 'take off a hat', 'cheer up', 'hand waving',
                        'kicking something', 'jump up', 'hopping', 'reach into pocket', 'make a phone call',
                        'playing with phone', 'typing on a keyboard', 'pointing to something with finger',
                        'taking a selfie', 'check time from watch', 'rub two hands together', 'nod head', 'shake head',
                        'wipe face', 'salute', 'put the palms together', 'cross hands in front', 'sneeze', 'staggering',
                        'falling', 'touch head (headache)', 'touch chest (heart pain)', 'touch back (backache)',
                        'touch neck (neckache)', 'nausea', 'use a fan', 'punching other person', 'kicking other person',
                        'pushing other person', 'pat on back of other person', 'point finger at the other person',
                        'hugging other person', 'giving something to other person', 'touch some person pocket',
                        'handshaking', 'sitting down', 'walking towards each other']

        self.frame = []
        self.video = []
        self.lmList = []


    '''程序界面布局'''

    def set_ui(self):
        self.__layout_main = QtWidgets.QHBoxLayout()  # 总布局
        self.__layout_fun_button = QtWidgets.QVBoxLayout()  # 按键布局
        self.__layout_data_show = QtWidgets.QVBoxLayout()  # 数据(视频)显示布局

        self.button_open_camera = QtWidgets.QPushButton('打开相机')  # 建立用于打开摄像头的按键
        self.button_upload_file = QtWidgets.QPushButton('上传视频')  # 建立用于上传视频
        self.button_close = QtWidgets.QPushButton('退出')  # 建立用于退出程序的按键
        self.button_open_camera.setMinimumHeight(50)  # 设置按键大小
        self.button_upload_file.setMinimumHeight(50)
        self.button_close.setMinimumHeight(50)

        self.button_close.move(10, 100)  # 移动按键
        '''信息显示'''
        self.label_result = QtWidgets.QLabel()  # 定义显示分类的Label
        self.label_result.setFixedSize(350,50)  # 给显示分类的Label设置大小为250x50
        self.label_result.setText("-")
        self.label_result.setStyleSheet("color:red;font:bold 25px;")
        self.label_result.setFont(QFont("Microsoft YaHei"))
        self.label_show_camera = QtWidgets.QLabel()  # 定义显示视频的Label
        self.label_show_camera.setFixedSize(641, 481)  # 给显示视频的Label设置大小为641x481
        '''把按键加入到按键布局中'''
        self.__layout_fun_button.addWidget(self.label_result)  # 把用于显示分类的Label加入到按键布局中
        self.__layout_fun_button.addWidget(self.button_open_camera)  # 把打开摄像头的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_upload_file)  # 把上传视频的按键放到按键布局中
        self.__layout_fun_button.addWidget(self.button_close)  # 把退出程序的按键放到按键布局中
        '''把某些控件加入到总布局中'''
        self.__layout_main.addLayout(self.__layout_fun_button)  # 把按键布局加入到总布局中
        self.__layout_main.addWidget(self.label_show_camera)  # 把用于显示视频的Label加入到总布局中
        '''总布局布置好后就可以把总布局作为参数传入下面函数'''
        self.setLayout(self.__layout_main)  # 到这步才会显示所有控件

    '''初始化所有槽函数'''

    def slot_init(self):
        self.button_open_camera.clicked.connect(
            self.button_open_camera_clicked)  # 若该按键被点击，则调用button_open_camera_clicked()
        self.button_upload_file.clicked.connect(self.button_upload_file_clicked)
        self.timer_camera.timeout.connect(self.show_camera)  # 若定时器结束，则调用show_camera()
        self.button_close.clicked.connect(self.close)  # 若该按键被点击，则调用close()，注意这个close是父类QtWidgets.QWidget自带的，会关闭程序

    '''槽函数之一'''

    def button_open_camera_clicked(self):
        if self.timer_camera.isActive() == False:  # 若定时器未启动
            flag = self.cap.open(self.CAM_NUM)  # 参数是0，表示打开笔记本的内置摄像头，参数是视频文件路径则打开视频
            if flag == False:  # flag表示open()成不成功
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(30)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示
                self.button_open_camera.setText('关闭相机')
        else:
            self.video = []
            self.frame = []
            self.lmList = []
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域
            self.button_open_camera.setText('打开相机')

    def button_upload_file_clicked(self):
        self.video = []
        self.frame = []
        self.lmList = []
        file = QFileDialog.getOpenFileName(self, "请选择文件", "C:\\Users\\86134\Desktop\\videPoroject\\ActionRecognize")
        print(file[0])
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(file[0])
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(self, 'warning', "请检查相机于电脑是否连接正确", buttons=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(1)  # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示

    def show_camera(self):
        try:
            flag, self.image = self.cap.read()  # 从视频流中读取
            self.image = self.detector.findPose(self.image) # 检测人体框架
            self.lmList = self.detector.findPostion(self.image) # 返回坐标
            for i in range(len(self.lmList)):
                if i in self.need:
                    for j in self.lmList[i]:
                        self.frame.append(j)
            self.video.append(self.frame)
            self.frame = []
            if len(self.video) == 100:
                df = pd.DataFrame(self.video)
                df = (df - df.min()) / (df.max() - df.min())
                self.video = np.array(df)
                self.video = self.video.reshape(1, 100, 57)
                # print(self.video)
                # print(self.video)
                pred = self.model.predict(self.video)
                self.video = []
                pred2 = np.argmax(pred, axis=1)
                # print(pred2[0])
                # print(self.action_names[pred2[0]])
                if self.action_names[pred2[0]] == "null":
                    self.label_result.setText('Please try again')
                else:
                    self.label_result.setText(self.action_names[pred2[0]])
                # print(pred[0][0])
            cTime = time.time()
            fps = 1/(cTime-self.pTime)
            self.pTime = cTime
            cv2.putText(self.image, str(int(fps)), (20, 70), cv2.FONT_HERSHEY_PLAIN,
                        3, (0, 255, 0), 3)
            show = cv2.resize(self.image, (640, 480))  # 把读到的帧的大小重新设置为 640x480
            show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)  # 视频色彩转换回RGB，这样才是现实的颜色
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0],
                                     QtGui.QImage.Format_RGB888)  # 把读取到的视频数据变成QImage形式
            self.label_show_camera.setPixmap(QtGui.QPixmap.fromImage(showImage))  # 往显示视频的Label里 显示QImage
        except:
            self.timer_camera.stop()  # 关闭定时器
            self.cap.release()  # 释放视频流
            self.label_show_camera.clear()  # 清空视频显示区域


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)  # 固定的，表示程序应用
    ui = Ui_MainWindow()  # 实例化Ui_MainWindow
    ui.show()  # 调用ui的show()以显示。同样show()是源于父类QtWidgets.QWidget的
    sys.exit(app.exec_())  # 不加这句，程序界面会一闪而过