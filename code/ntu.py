"""
Author: Henry Powell
Institution: Institute of Neuroscience and Psychology, Glasgow University, Scotland.

Python script for formatting the NTU RGB+D Skeletons data set into a format suitable for most LSTM RNNs. The aim is to
take each .skeletons file and compress it into a 3D numpy array with [samples, time-steps, features] as its dimensions.
The final data set will thus be a [56,881, max(len(samples(data_files)))=600, 12*25=300] numpy array. The data has
been left normal (i.e. not normalized) for the sake of flexibility although it is generally recommended to normalize
the data at some stage in the preprocessing.
"""
# 设置断点的时候，注意挂起这一项应该选ALL，意思是任何情况下代码运行到断点时都要暂停运行进入等待状态

from __future__ import print_function

import numpy as np
import pandas as pd
import gc
from pathlib import Path
import tensorflow as tf
import random
import time
# import os
# import cupy as cp


np_cp = np
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Insert path to .skeleton files here

# 需要注意path的形式，应该是本工程目录的源数据集目录的字符串加上斜杆
path = 'dataset/'
# dest_path = './processed_dataset'
# dest_directory = Path(dest_path)

# Keep track of total files processed
total_files = 0

# List of class numbers labels from data_set
NTU_classes = [c for c in range(1, 61)]


def filter_missing_samples():
    """ Function to filter out all of the samples from the data set that have no data in them.

        Returns: list object containing str(filenames) of all files with no data

        函数从数据集中过滤出没有数据的所有样本。返回:列表对象包含所有没有数据的文件的str(文件名)
    """


    begin = time.time()
    # List of files with missing data.
    missing = ['S001C002P005R002A008', 'S001C002P006R001A008', 'S001C003P002R001A055', 'S001C003P002R002A012',
               'S001C003P005R002A004', 'S001C003P005R002A005', 'S001C003P005R002A006', 'S001C003P006R002A008',
               'S002C002P011R002A030', 'S002C003P008R001A020', 'S002C003P010R002A010', 'S002C003P011R002A007',
               'S002C003P011R002A011', 'S002C003P014R002A007', 'S003C001P019R001A055', 'S003C002P002R002A055',
               'S003C002P018R002A055', 'S003C003P002R001A055', 'S003C003P016R001A055', 'S003C003P018R002A024',
               'S004C002P003R001A013', 'S004C002P008R001A009', 'S004C002P020R001A003', 'S004C002P020R001A004',
               'S004C002P020R001A012', 'S004C002P020R001A020', 'S004C002P020R001A021', 'S004C002P020R001A036',
               'S005C002P004R001A001', 'S005C002P004R001A003', 'S005C002P010R001A016', 'S005C002P010R001A017',
               'S005C002P010R001A048', 'S005C002P010R001A049', 'S005C002P016R001A009', 'S005C002P016R001A010',
               'S005C002P018R001A003', 'S005C002P018R001A028', 'S005C002P018R001A029', 'S005C003P016R002A009',
               'S005C003P018R002A013', 'S005C003P021R002A057', 'S006C001P001R002A055', 'S006C002P007R001A005',
               'S006C002P007R001A006', 'S006C002P016R001A043', 'S006C002P016R001A051', 'S006C002P016R001A052',
               'S006C002P022R001A012', 'S006C002P023R001A020', 'S006C002P023R001A021', 'S006C002P023R001A022',
               'S006C002P023R001A023', 'S006C002P024R001A018', 'S006C002P024R001A019', 'S006C003P001R002A013',
               'S006C003P007R002A009', 'S006C003P007R002A010', 'S006C003P007R002A025', 'S006C003P016R001A060',
               'S006C003P017R001A055', 'S006C003P017R002A013', 'S006C003P017R002A014', 'S006C003P017R002A015',
               'S006C003P022R002A013', 'S007C001P018R002A050', 'S007C001P025R002A051', 'S007C001P028R001A050',
               'S007C001P028R001A051', 'S007C001P028R001A052', 'S007C002P008R002A008', 'S007C002P015R002A055',
               'S007C002P026R001A008', 'S007C002P026R001A009', 'S007C002P026R001A010', 'S007C002P026R001A011',
               'S007C002P026R001A012', 'S007C002P026R001A050', 'S007C002P027R001A011', 'S007C002P027R001A013',
               'S007C002P028R002A055', 'S007C003P007R001A002', 'S007C003P007R001A004', 'S007C003P019R001A060',
               'S007C003P027R002A001', 'S007C003P027R002A002', 'S007C003P027R002A003', 'S007C003P027R002A004',
               'S007C003P027R002A005', 'S007C003P027R002A006', 'S007C003P027R002A007', 'S007C003P027R002A008',
               'S007C003P027R002A009', 'S007C003P027R002A010', 'S007C003P027R002A011', 'S007C003P027R002A012',
               'S007C003P027R002A013', 'S008C002P001R001A009', 'S008C002P001R001A010', 'S008C002P001R001A014',
               'S008C002P001R001A015', 'S008C002P001R001A016', 'S008C002P001R001A018', 'S008C002P001R001A019',
               'S008C002P008R002A059', 'S008C002P025R001A060', 'S008C002P029R001A004', 'S008C002P031R001A005',
               'S008C002P031R001A006', 'S008C002P032R001A018', 'S008C002P034R001A018', 'S008C002P034R001A019',
               'S008C002P035R001A059', 'S008C002P035R002A002', 'S008C002P035R002A005', 'S008C003P007R001A009',
               'S008C003P007R001A016', 'S008C003P007R001A017', 'S008C003P007R001A018', 'S008C003P007R001A019',
               'S008C003P007R001A020', 'S008C003P007R001A021', 'S008C003P007R001A022', 'S008C003P007R001A023',
               'S008C003P007R001A025', 'S008C003P007R001A026', 'S008C003P007R001A028', 'S008C003P007R001A029',
               'S008C003P007R002A003', 'S008C003P008R002A050', 'S008C003P025R002A002', 'S008C003P025R002A011',
               'S008C003P025R002A012', 'S008C003P025R002A016', 'S008C003P025R002A020', 'S008C003P025R002A022',
               'S008C003P025R002A023', 'S008C003P025R002A030', 'S008C003P025R002A031', 'S008C003P025R002A032',
               'S008C003P025R002A033', 'S008C003P025R002A049', 'S008C003P025R002A060', 'S008C003P031R001A001',
               'S008C003P031R002A004', 'S008C003P031R002A014', 'S008C003P031R002A015', 'S008C003P031R002A016',
               'S008C003P031R002A017', 'S008C003P032R002A013', 'S008C003P033R002A001', 'S008C003P033R002A011',
               'S008C003P033R002A012', 'S008C003P034R002A001', 'S008C003P034R002A012', 'S008C003P034R002A022',
               'S008C003P034R002A023', 'S008C003P034R002A024', 'S008C003P034R002A044', 'S008C003P034R002A045',
               'S008C003P035R002A016', 'S008C003P035R002A017', 'S008C003P035R002A018', 'S008C003P035R002A019',
               'S008C003P035R002A020', 'S008C003P035R002A021', 'S009C002P007R001A001', 'S009C002P007R001A003',
               'S009C002P007R001A014', 'S009C002P008R001A014', 'S009C002P015R002A050', 'S009C002P016R001A002',
               'S009C002P017R001A028', 'S009C002P017R001A029', 'S009C003P017R002A030', 'S009C003P025R002A054',
               'S010C001P007R002A020', 'S010C002P016R002A055', 'S010C002P017R001A005', 'S010C002P017R001A018',
               'S010C002P017R001A019', 'S010C002P019R001A001', 'S010C002P025R001A012', 'S010C003P007R002A043',
               'S010C003P008R002A003', 'S010C003P016R001A055', 'S010C003P017R002A055', 'S011C001P002R001A008',
               'S011C001P018R002A050', 'S011C002P008R002A059', 'S011C002P016R002A055', 'S011C002P017R001A020',
               'S011C002P017R001A021', 'S011C002P018R002A055', 'S011C002P027R001A009', 'S011C002P027R001A010',
               'S011C002P027R001A037', 'S011C003P001R001A055', 'S011C003P002R001A055', 'S011C003P008R002A012',
               'S011C003P015R001A055', 'S011C003P016R001A055', 'S011C003P019R001A055', 'S011C003P025R001A055',
               'S011C003P028R002A055', 'S012C001P019R001A060', 'S012C001P019R002A060', 'S012C002P015R001A055',
               'S012C002P017R002A012', 'S012C002P025R001A060', 'S012C003P008R001A057', 'S012C003P015R001A055',
               'S012C003P015R002A055', 'S012C003P016R001A055', 'S012C003P017R002A055', 'S012C003P018R001A055',
               'S012C003P018R001A057', 'S012C003P019R002A011', 'S012C003P019R002A012', 'S012C003P025R001A055',
               'S012C003P027R001A055', 'S012C003P027R002A009', 'S012C003P028R001A035', 'S012C003P028R002A055',
               'S013C001P015R001A054', 'S013C001P017R002A054', 'S013C001P018R001A016', 'S013C001P028R001A040',
               'S013C002P015R001A054', 'S013C002P017R002A054', 'S013C002P028R001A040', 'S013C003P008R002A059',
               'S013C003P015R001A054', 'S013C003P017R002A054', 'S013C003P025R002A022', 'S013C003P027R001A055',
               'S013C003P028R001A040', 'S014C001P027R002A040', 'S014C002P015R001A003', 'S014C002P019R001A029',
               'S014C002P025R002A059', 'S014C002P027R002A040', 'S014C002P039R001A050', 'S014C003P007R002A059',
               'S014C003P015R002A055', 'S014C003P019R002A055', 'S014C003P025R001A048', 'S014C003P027R002A040',
               'S015C001P008R002A040', 'S015C001P016R001A055', 'S015C001P017R001A055', 'S015C001P017R002A055',
               'S015C002P007R001A059', 'S015C002P008R001A003', 'S015C002P008R001A004', 'S015C002P008R002A040',
               'S015C002P015R001A002', 'S015C002P016R001A001', 'S015C002P016R002A055', 'S015C003P008R002A007',
               'S015C003P008R002A011', 'S015C003P008R002A012', 'S015C003P008R002A028', 'S015C003P008R002A040',
               'S015C003P025R002A012', 'S015C003P025R002A017', 'S015C003P025R002A020', 'S015C003P025R002A021',
               'S015C003P025R002A030', 'S015C003P025R002A033', 'S015C003P025R002A034', 'S015C003P025R002A036',
               'S015C003P025R002A037', 'S015C003P025R002A044', 'S016C001P019R002A040', 'S016C001P025R001A011',
               'S016C001P025R001A012', 'S016C001P025R001A060', 'S016C001P040R001A055', 'S016C001P040R002A055',
               'S016C002P008R001A011', 'S016C002P019R002A040', 'S016C002P025R002A012', 'S016C003P008R001A011',
               'S016C003P008R002A002', 'S016C003P008R002A003', 'S016C003P008R002A004', 'S016C003P008R002A006',
               'S016C003P008R002A009', 'S016C003P019R002A040', 'S016C003P039R002A016', 'S017C001P016R002A031',
               'S017C002P007R001A013', 'S017C002P008R001A009', 'S017C002P015R001A042', 'S017C002P016R002A031',
               'S017C002P016R002A055', 'S017C003P007R002A013', 'S017C003P008R001A059', 'S017C003P016R002A031',
               'S017C003P017R001A055', 'S017C003P020R001A059', 'S001C002P006R001A008']

    missing_skeleton = [path + i + '.skeleton' for i in missing]
    missing = missing_skeleton
    del missing_skeleton
    gc.collect()  # 清除内存
    end = time.time()
    print('过滤没有数据的样本使用的时间: {} \n'.format(end-begin))
    return missing


def load_files(path, missing, beginning_file_num=0, ending_file_num=30000, training_ratio=0.8, batch_type='train', drop_first=True):

    """
    :param path: Path to the data set.
    :param missing: List of files with no data.
    :param beginning_file_num: 第一个需要处理的文件的序号。从0开始计数。
    :param ending_file_num: 最后一个需要处理的文件的序号；如果最后一个需要处理的文件为所有有效文件的最后一个，那么设置为0。从0开始计数。
    :param training_ratio: What proportion of the fix_total_files you want to load.一般是0.8或者0.7
    :param batch_type: Splits the loaded files into either 80% of total files if batch_type = 'train', or 20% of
                       total files if batch_type = 'test'.
    :param drop_first: Stop function from iterating over .CD file if there is one present in the directory.
    :return: List of .skeleton files as posixpath objects
    """
    begin = time.time()
    directory = Path(path)

    # Store files as list to be iterated through

    # directory.iterdir()按照某种顺序读取目录中的文件
    files = [p for p in directory.iterdir() if p.is_file() and str(p) not in missing]

    # You may have a .CD file hidden in this folder. This drops this from [files] so that the code doesn't run over it.
    if drop_first:
        files.pop(0)
    else:
        pass

    if not ending_file_num:
        files = files[beginning_file_num:]
    else:
        files = files[beginning_file_num:ending_file_num+1]

    # Number of total files before dropping if files_batch_prop < 100
    # total_num_files = len(files)
    file_num_for_train = (ending_file_num-beginning_file_num+1) * training_ratio

    # Drop proportion of files you don't want to process
    if batch_type == 'train':
        files = files[:int(file_num_for_train)]
    elif batch_type == 'test':
        files = files[int(file_num_for_train):]
    # elif prop_files > 100 or prop_files < 0:
    #     raise Exception('files_batch_prop should be an integer between 0 and 100. You gave {}'.format(prop_files))
    gc.collect()
    end = time.time()
    print('加载数据使用的时间: {} \n'.format(end - begin))

    return files


def get_classes(files, one_hot=False):

    """

    :param files: list of .skeleton files to be processed (must be posixPath object)
    :param one_hot: translate classes to a one-hot encoding
    # :param subset: specify that you are using a the binary subset of the dataset (Action 1 and Action 3 (A001 & A003)
    :return: list of classes

    """
    begin = time.time()

    files = [str(f) for f in files]
    class_list = list()
    class_index = files[0].find('A0')

    for i in range(len(files)):
        data = pd.read_csv(files[i], header=None)
        try:
            data['length'] = data[0].apply(lambda x: len(x))  # 获取每行数据的长度
            class_list.append(files[i][class_index + 2:class_index + 4])
        except:
            continue
        # class_list.append(files[i][class_index+2:class_index+4])
    print("--------------------------------------")
    print(len(class_list))
    print("--------------------------------------")
    del class_index

    class_list = [int(c)-1 for c in class_list]

    # # Delete below when not using 2 class version of the data set
    # class_list = [int(c/2) for c in class_list]

    class_list = np_cp.array(class_list)

    if one_hot:
        # One-hot encode integers to make suitable for LSTM
        class_list = tf.keras.utils.to_categorical(class_list)

    else:
        pass

    gc.collect()
    end = time.time()
    print('获取标签使用的时间: {} \n'.format(end - begin))
    return class_list

def ndarray_nearest_neighbour_scaling(label, new_h, new_w):
    """
    Implement nearest neighbour scaling for ndarray
    :param label: [H, W] or [H, W, C]
    :return: label_new: [new_h, new_w] or [new_h, new_w, C]
    Examples
    --------
    ori_arr = np.array([[1, 2, 3],
                        [4, 5, 6],
                        [7, 8, 9]], dtype=np.int32)
    new_arr = ndarray_nearest_neighbour_scaling(ori_arr, new_h=8, new_w=10)
    >> print(new_arr)
    [[1 1 1 1 2 2 2 3 3 3]
     [1 1 1 1 2 2 2 3 3 3]
     [1 1 1 1 2 2 2 3 3 3]
     [4 4 4 4 5 5 5 6 6 6]
     [4 4 4 4 5 5 5 6 6 6]
     [4 4 4 4 5 5 5 6 6 6]
     [7 7 7 7 8 8 8 9 9 9]
     [7 7 7 7 8 8 8 9 9 9]]
    """
    if len(label.shape) == 2:
        label_new = np_cp.zeros([new_h, new_w], dtype=label.dtype)
    else:
        label_new = np_cp.zeros([new_h, new_w, label.shape[2]], dtype=label.dtype)

    scale_h = new_h / label.shape[0]
    scale_w = new_w / label.shape[1]

    y_pos = np_cp.arange(new_h)
    x_pos = np_cp.arange(new_w)
    y_pos = np_cp.floor(y_pos / scale_h).astype(np_cp.int32)
    x_pos = np_cp.floor(x_pos / scale_w).astype(np_cp.int32)

    y_pos = y_pos.reshape(y_pos.shape[0], 1)
    y_pos = np_cp.tile(y_pos, (1, new_w))
    x_pos = np_cp.tile(x_pos, (new_h, 1))
    assert y_pos.shape == x_pos.shape

    label_new[:, :] = label[y_pos[:, :], x_pos[:, :]]
    return label_new


def process_raw_data(files, save_as_ndarray=False, three_d=True, derivative=False):

    """
    :param files: list of .skeleton files to be processed (must be posixPath object)
    :param save_as_ndarray: set to True to save the outputted data to an ndarray in the current directory
    :param derivative: add feature engineered columns to the output. Adds first derivative calculations to each
                       position point in x,y,z dimensions.
    :param three_d: set to True if you only want the three d position features for each time frame
    :return: np.array of dimension (samples, time_steps, features)

    """

    # This variable tracks how many files have been formatted and added to the new data set
    progress = 0
    loaded = list()
    # frames_len = list()
    # interpolation_frames = np_cp.linspace(0, 1, 100)
    begin = time.time()
    # Iteration loop which formats the .skeleton files.
    for file in files:
        features = list()
        # row = list()


        data = pd.read_csv(file, header=None)
        # frames_len_temp = int(data[0][0])
        if progress >= 172:
            print('当前已经处理的文件数量为: {} \n'.format(progress))
            print(data[0].shape)

        try:
            data['length'] = data[0].apply(lambda x: len(x))  # 获取每行数据的长度
        except:
            print(file)
            continue
        cond = data['length'] > 10  # 只保留25个节点的坐标和深度信息等
        data = data[cond]
        data = data.reset_index(drop=True)
        data = data[data.index % 26 != 0]
        # 这一行代码意思是去掉'bodyID', 'clipedEdges', 'handLeftConfidence','handLeftState', 'handRightConfidence',
        # 'handRightState','isResticted', 'leanX', 'leanY', 'trackingState'等数据

        data = data.drop(columns=['length'])
        data = data.reset_index(drop=True)
        data = data[0].str.split(" ", expand=True)  # 数据分割，分列的依据是空格；expand=True表示将分割出来的数据当成一列
        data = data.fillna(method='bfill')  # 空值数据填充：用相邻特征填充空值
        if three_d:
            data = data.drop(columns=[3, 4, 5, 6, 7, 8, 9, 10, 11])
            data = data.reset_index(drop=True)
            '''
            0到11共12个数据，分别代表'x', 'y', 'z', 'depthX', 'depthY', 'colorX', 
            'colorY','orientationW', 'orientationX', 'orientationY','orientationZ', 'trackingState'

            '''

        if derivative:
            x_pos, y_pos, z_pos = np_cp.array(data[0], dtype=np_cp.float32), \
                                  np_cp.array(data[1], dtype=np_cp.float32), \
                                  np_cp.array(data[2], dtype=np_cp.float32)

            data[3], data[4], data[5] = pd.Series(np_cp.gradient(x_pos)), \
                                        pd.Series(np_cp.gradient(y_pos)), \
                                        pd.Series(np_cp.gradient(z_pos))

        frames = int(len(data.index) / 25)
        # frames_len.append(frames)

        # Make features array
        for j in range(len(data)):
            features.append(data.iloc[j])  # 得到属性名、数据和数据类型
        del data
        gc.collect()

        features = np_cp.array(features, dtype=np_cp.float16)
        # 转成ndarray格式，如array([[1,2],[3,4],[5,6]])；dtype=np_cp.float16指定数据类型
        features = features.flatten()  # 将array格式折叠成一个一维的数组
        features = np_cp.array(np_cp.split(features, frames))
        # np_cp.save('./np_cpytest/rowtest1.npy', row)
        # 用frames（整数）将数组row平均切分

        # row = row.tolist()  # 转成列表
        #
        # features.append(row)
        # del row
        # gc.collect()
        # features = np_cp.array(features)
        # features = features[0]
        # np_cp.save('./npytest/features1test1', features)

        # 插值或者下采样
        # features_stack = np_cp.stack(features, axis=1)
        # print(features.shape)
        if frames < 100:
            features = ndarray_nearest_neighbour_scaling(features, 100, 75)

            # # 下面这种一维的插值方法不能用，因为对于每一列，插入的帧的位置不同
            # org_frames = np_cp.linspace(0, 1, frames)
            # features_interpolation = list()
            # features = np_cp.stack(features, axis=1)
            # for i in range(75):
            #     features_interpolation_temp = np_cp.interp(interpolation_frames, org_frames, features[i])
            #     features_interpolation.append(features_interpolation_temp)
            # features = features_interpolation

        elif frames > 100:
            sample_frames = [i for i in range(frames)]
            sample_frames = random.sample(sample_frames, 100)
            sample_frames.sort()
            features = features[sample_frames]


        # # 下面是原代码的方法
        # mid = len(features[0])//2
        # start = mid - 15
        # end = mid + 15
        # features = features[0][start:end]
        # np_cp.save('./npytest/features2test1', features)


        # 中心化，用其他关节点的坐标减去中心关节（2）的坐标
        features = np_cp.array(features)
        # features = features.flatten()
        # features = np_cp.split(features, 2500)  # features现在是list:2500，每个元素是array，每个array包含xyz
        # center_joint = features[100:200]
        # center_joint = np_cp.tile(center_joint, (25, 1))
        # features = np_cp.array(features)
        # features = features - center_joint
        loaded.append(features)
        if progress == 200:
            end = time.time()
            print('处理20个file使用的时间: {} \n'.format(end-begin))


        # if save_as_ndarray:
        #     np_cp.save(os.path.join(dest_path, str(file)), features)

        # Sanity check to ensure all the matrices are of the right dimension (Uncomment the below to make check)
        # print(features.shape)
        #
        # This block tracks the progress of the formatting and prints the progress to the terminal.
        # The "end='\r' argument has only been tested on macOS and may not work on other platforms.
        # If you don't see any progress on your terminal delete this.
        if progress % 10 == 0:
            print('当前已经处理的文件数量为: {} \n'.format(progress))
        progress += 1
        # perc = progress/(56881/100)

        # # Save the array to a new CSV named "skeleton_data_formatted.csv"
        # print('Samples Processed: {0}/56,881 - - - Percentage Complete = {1:.2f}%'.format(progress, perc), end='\r')
        #
        # if progress == total_files:
        #     print('Samples Processed: {0}/56,881 - - - Percentage Complete = {1:.2f}%'.format(progress, 100))



    # # Stack matrices together in 3rd dimension (features)
    # # loaded0 = loaded[0]
    # # loaded0 = np_cp.array(loaded0)
    # # np_cp.save('loaded0test1.npy', loaded0)
    # frames_len = np_cp.array(frames_len)
    # np_cp.save('frames_len.npy', frames_len)
    # # loaded = np_cp.squeeze(loaded, axis=0)

    # 下面是调试的代码
    # print(type(loaded))
    # print(loaded.shape)

    # loaded = np_cp.stack(loaded, axis=0)

    loaded = np_cp.array(loaded)


    # np_cp.save(os.path.join(dest_path, str(file)), loaded)
    return loaded


def preprocess_training(beginning_file_num=0, ending_file_num=30000, training_ratio=0.8, sanity=False, save=True, one_hot=False):

    print('Processing Training Set')

    # missing, files, classes&loaded是递进的关系
    missing = filter_missing_samples()
    files = load_files(path, missing, beginning_file_num=beginning_file_num, ending_file_num=ending_file_num,
                       training_ratio=training_ratio,
                       batch_type='train')

    # classes和loaded是并行的关系
    classes = get_classes(files, one_hot=one_hot)
    loaded = process_raw_data(files, save_as_ndarray=False, three_d = True, derivative = False)
    if save:
        np_cp.save('skeletons_array_train_S.npy', loaded)
        np_cp.save('skeletons_array_train_labels_S.npy', classes)
    else:
        pass

    # Sanity check to ensure resulting matrix is of the right shape
    print('Final training data dimensions: {}'.format(loaded.shape))

    if sanity:
        print('One-hot classes matrix:\n', classes)
    else:
        pass

    print('Final training labels dimensions: {} \n'.format(classes.shape))

    return loaded, classes


def preprocess_test(beginning_file_num=1250, ending_file_num=30000, training_ratio=0.8, sanity=False, save=True):

    print('Processing Test Set')
    missing = filter_missing_samples()
    files = load_files(path, missing, beginning_file_num, ending_file_num, training_ratio,
                       batch_type='test')
    classes = get_classes(files)
    loaded = process_raw_data(files, save_as_ndarray=True, three_d=True, derivative=False)
    if save:
        np_cp.save('skeletons_array_test_S.npy', loaded)
        np_cp.save('skeletons_array_test_labels_S.npy', classes)
    else:
        pass

    # Sanity check to ensure resulting matrix is of the right shape
    print('Final Training data dimensions: {}'.format(loaded.shape))

    if sanity:
        print('One-hot classes matrix:\n', classes)
    else:
        pass

    print('Final test labels dimensions: {} \n'.format(classes.shape))

    return loaded, classes


def get_test_train(beginning_file_num=0, ending_file_num=30000, training_ratio=0.8, sanity=False, save=True):

    preprocess_training(beginning_file_num=beginning_file_num, ending_file_num=ending_file_num, training_ratio=training_ratio, sanity=sanity, save=save)
    preprocess_test(beginning_file_num=beginning_file_num, ending_file_num=ending_file_num, training_ratio=training_ratio, sanity=sanity, save=save)


# def get_partition_labels(files):
#
#     files = [str(f) for f in files]
#     train_list = files[:45505]
#     validation_list = files[45505:]
#
#     classes = get_classes(files, one_hot=False)
#     classes = [c+1 for c in classes]
#
#     partition = {'train': train_list,
#                  'validation': validation_list}
#
#     labels = dict(zip(files, classes))
#
#     return partition, labels


get_test_train(beginning_file_num=0, ending_file_num=56800, training_ratio=0.8, sanity=False, save=True)
