import pandas as pd  # 引用套件並縮寫為 pd
import csv
import numpy as np
import os
import math
import sys
from datetime import datetime
from time import strftime
import numpy

numpy.set_printoptions(threshold=numpy.nan)
np.set_printoptions(formatter={'float': '{: 0.1f}'.format})


def sigmoid(x):
  res =  1 / (1.0 + np.exp(-x))
  return np.clip(res, 0.00000000000001, 0.99999999999999)


batch_size = 25
W = np.zeros((106, ))       #權重
B = np.zeros((1, ))        #Bias
Z = np.ones((32561, ))
y = np.ones((32561, ))
y_ad = np.ones((32561, ))
Y = np.ones((32561, ))
l_rate = 10**-10# + 0.00000000015
Training_T = 50000
Buf = 0
B_correct = 0

gra = np.zeros((106,))
prev_gra = np.zeros((106,))
ada = np.zeros((106,))

def train():
    batch_size = 25
    W = np.zeros((106,))  # 權重
    B = np.zeros((1,))  # Bias
    Z = np.ones((32561,))
    y = np.ones((32561,))
    y_ad = np.ones((32561,))
    Y = np.ones((32561,))
    l_rate = 10 ** -10  # + 0.00000000015
    Training_T = 1000
    Buf = 0
    B_correct = 0

    gra = np.zeros((106,))
    prev_gra = np.zeros((106,))
    ada = np.zeros((106,))
    # =====================X_train.txt=====================================
    X_train = open('X_train.txt', 'r', newline="")
    line = X_train.readlines()
    X_train_Data_32561_106 = np.ones((32561, 106))

    C = 0
    S = 0
    for i in line:
        # print (i)
        if S != 0:
            X_train_Data_32561_106[C] = i.strip().split(',')
            # print(Data_32562_15[C])
            C = C + 1
        S = S + 1

    min = np.amin(X_train_Data_32561_106, axis=0)
    #print("min : ", min[0])
    MAX = np.amax(X_train_Data_32561_106, axis=0)
    #print("MAX : ", MAX[0])
    X_train_Data_32561_106 = (X_train_Data_32561_106 - min) / (MAX - min)

    # print(X_train_Data_32562_106[32560])
    X_train.close()
    # =====================================================================



    # ==========================Y_train====================================
    Y_train = open('Y_train.txt', 'r', newline="")
    line = Y_train.readlines()
    Y_train_Data_32561_1 = np.ones((32561,))

    C = 0
    S = 0
    for i in line:
        # print (i)
        if S != 0:
            Y_train_Data_32561_1[C] = i
            # Y_train_Data_32561_1[C] = i.strip().split(',')
            # print(Data_32562_15[C])
            C = C + 1
        S = S + 1
    # print(Y_train_Data_32561_1[32560])
    Y_train.close()
    # =====================================================================



    for i in range(Training_T):
        Z = np.dot(X_train_Data_32561_106, W) + B
        y = sigmoid(Z)

        cross_entropy = -(np.dot(np.transpose(Y_train_Data_32561_1), np.log(y)) + np.dot(np.transpose(1 - Y_train_Data_32561_1), np.log(1 - y)))

        #print("T : ",i,"  Cross : ",cross_entropy)
        #print(cross_entropy)

        #=================加權值訓練============================================
        W_X_train_Data_25_106 = np.zeros((25, 106))
        W_Y_train_Data_25_1 = np.zeros((25, 1))
        for A in range(25):
            W_X_train_Data_25_106[A] = X_train_Data_32561_106[A + Buf]
            W_Y_train_Data_25_1[A] = (Y_train_Data_32561_1 - y)[A + Buf]

        #print(W_X_train_Data_25_106)
        '''''
        W_Y_train_Data_25_1 = np.zeros((25, 1))
        for B in range(25):
            W_Y_train_Data_25_1[B] = (Y_train_Data_32561_1-y)[B + Buf]
        '''''
        #print(Y_train_Data_32561_1-y)

        w_grad = np.sum(-1* W_X_train_Data_25_106 * W_Y_train_Data_25_1.reshape((batch_size,1)),axis = 0)

        l_rate = 8 * 10 * -3 / (1 + i) * 0.05
        W = W - l_rate * w_grad
        #=======================================================================

        #===================Bias訓練============================================
        b_grad = np.sum(-1 * (W_Y_train_Data_25_1))
        B = B - l_rate * b_grad
        #=======================================================================

        # ===================Adargrad=========================================

        y[y >= 0.5] = 1
        y[y < 0.5] = 0

        '''''
        L = y + Y_train_Data_32561_1
        gra = 2 * np.dot(X_train_Data_32561_106.T, L)
        prev_gra += gra ** 2
        ada = np.sqrt(prev_gra)
        W -= (l_rate * gra) / ada  # 使用 adagrad優化
        '''''
        # =======================================================================

        count = 0
        if i%1000 == 0:
            print(i)
            for a in range(32561):
                if y[a] == Y_train_Data_32561_1[a]:
                    count = count + 1

            print("Cross : ",cross_entropy)
            print("Correct : ",count / 32561)


        Buf = Buf + 25
        if Buf > 32535:
            Buf = 0


    print("Training Complete")

    '''''
    with open('W.csv', 'w', newline="") as csv_file:  # 開 W 加權值 _寫檔
        w_csv_writer = csv.writer(csv_file)
        for i in range(106):
            print(W[i])
            w_csv_writer.writerow(str(W[i]))
        csv_file.close()
    '''''

    W_wirte = open("W.txt", "w")
    for i in range(106):
        W_wirte.writelines(str(W[i]))
        W_wirte.write("\n")
    W_wirte.close()

    '''''
    with open('B.csv', 'w', newline="") as csv_file:  # 開 B _寫檔
        B_csv_writer = csv.writer(csv_file)
        for i in range(1):
            # print(W[i])
            B_csv_writer.writerow(str(B[i]))
        csvfile.close()

    print("File_W, File_B wirte out Complete")
    '''''

    B_wirte = open("B.txt", "w")
    for i in range(1):
        B_wirte.write(str(B[i]))
    B_wirte.close()


def test(file):
    # ===============================================================================
    with open('W.csv', 'r', newline="") as csvfile:  # 讀取 w(加權檔案) 的值
        w_reader = csv.reader(csvfile)
        W = np.ones((106, ))  # 加權值 w
        Count11 = 0
        for D in w_reader:
            W[Count11] = D
            Count11 = Count11 + 1
        # print(W)
    csvfile.close()
    # ================================================================================

    # ===============================================================================
    with open('B.csv', 'r', newline="") as csvfile:  # 讀取 w(加權檔案) 的值
        B_reader = csv.reader(csvfile)
        B = np.ones((1,))  # 加權值 w
        Count11 = 0
        for D in B_reader:
            B[Count11] = D
            Count11 = Count11 + 1
            # print(W)
    csvfile.close()
    # ================================================================================




    # ======================X_test.txt=====================================
    X_test = open(file, 'r', newline="")
    line = X_test.readlines()
    X_test_Data_16281_106 = np.ones((16281, 106))

    C = 0
    S = 0
    for i in line:
        # print (i)
        if S != 0:
            X_test_Data_16281_106[C] = i.strip().split(',')
            # print(Data_32562_15[C])
            C = C + 1
        S = S + 1
    # print(X_test_Data_16281_106[16280])

    min2 = np.amin(X_test_Data_16281_106, axis=0)
    #print("min : ", min[0])
    MAX2 = np.amax(X_test_Data_16281_106, axis=0)
    #print("MAX : ", MAX[0])
    X_test_Data_16281_106 = (X_test_Data_16281_106 - min2) / (MAX2 - min2)

    X_test.close()
    # =====================================================================


    Z = np.dot(X_test_Data_16281_106, W) + B
    y = sigmoid(Z)
    #print(Z)
    #print(y)
    #print(Z)

    ans_f = open("predictions.csv", mode = "w+")
    ans_f.write("id,label\n")
    G = 0
    for i in range(len(y)):
        j = i + 1
        A = 0
        if y[i] >= 0.5:
            G = G + 1
            A = 1
        ans_f.write(str(j) + "," + str(A) + "\n")
    ans_f.close()

    print(G)


if "_main_" == _name_:
    #print('0 : ',sys.argv[0])
    #print('1 : ',sys.argv[1])
    #print('2 : ',sys.argv[2])
    #print('3 : ',sys.argv[3])
    #print('4 : ',sys.argv[4])

    if sys.argv[2] == 'train':
        train()
    if sys.argv[2] == 'test':
        test(sys.argv[4])