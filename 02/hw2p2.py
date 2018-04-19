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
np.set_printoptions(formatter={'float': '{: 0.15f}'.format})


def sigmoid(x):
  res =  1 / (1.0 + np.exp(-x))
  return np.clip(res, 0.00000000000001, 0.99999999999999)



def train():
    #==============================下面開檔區==================================
    #=====================X_train.txt======================================
    X_train =  open('X_train.txt','r',newline="")
    line = X_train.readlines()
    X_train_Data_32561_106 = np.ones((32561, 106))

    C = 0
    S = 0
    for i in line:
        #print (i)
        if S != 0:
            X_train_Data_32561_106[C] = i.strip().split(',')
            #print(Data_32562_15[C])
            C = C + 1
        S = S + 1

    min = np.amin(X_train_Data_32561_106, axis = 0)
    #print("min : ",min[0])
    MAX = np.amax(X_train_Data_32561_106, axis = 0)
    #print("MAX : ",MAX[0])
    X_train_Data_32561_106 = (X_train_Data_32561_106 - min) / (MAX - min)


    #print(X_train_Data_32562_106[32560])
    X_train.close()
    #=====================================================================

    #==========================Y_train====================================
    Y_train =  open('Y_train.txt','r',newline="")
    line = Y_train.readlines()
    Y_train_Data_32561_1 = np.ones((32561,))

    C = 0
    S = 0
    for i in line:
        #print (i)
        if S != 0:
            Y_train_Data_32561_1[C] = i
            #Y_train_Data_32561_1[C] = i.strip().split(',')
            #print(Data_32562_15[C])
            C = C + 1
        S = S + 1
    #print(Y_train_Data_32561_1[32560])
    Y_train.close()
    #=====================================================================
    #==============================上面開檔區==================================

    #==============================下面變數宣告==================================
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
    #==============================上面變數宣告==================================

    for i in range(Training_T):
        Z = np.dot(X_train_Data_32561_106, W) + B
        y = sigmoid(Z)

        cross_entropy = -(np.dot(np.transpose(Y_train_Data_32561_1), np.log(y)) + np.dot(np.transpose(1 - Y_train_Data_32561_1), np.log(1 - y)))


        #=================加權值訓練============================================
        W_X_train_Data_25_106 = np.zeros((25, 106))
        W_Y_train_Data_25_1 = np.zeros((25, 1))
        for A in range(25):
            W_X_train_Data_25_106[A] = X_train_Data_32561_106[A + Buf]
            W_Y_train_Data_25_1[A] = (Y_train_Data_32561_1 - y)[A + Buf]

        #一次抓25比訓練

        w_grad = np.sum(-1* W_X_train_Data_25_106 * W_Y_train_Data_25_1.reshape((batch_size,1)),axis = 0)

        l_rate = 8 * 10 ** -3 / (1 + i) ** 0.05
        #使用adagard修改learning rate
        W = W - l_rate * w_grad
        #=======================================================================

        #===================Bias訓練============================================
        b_grad = np.sum(-1 * (W_Y_train_Data_25_1))
        B = B - l_rate * b_grad
        #=======================================================================


        #處理 sigmoid 後的 y
        y[y >= 0.5] = 1
        y[y < 0.5] = 0


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

    M_wirte = open("MAX.txt", "w")
    for U in range(106):
        M_wirte.write(str(MAX[U]))
        M_wirte.write("\n")
    M_wirte.close()

    m_wirte = open("min.txt", "w")
    for U2 in range(106):
        m_wirte.write(str(min[U2]))
        m_wirte.write("\n")
    m_wirte.close()

    W_wirte = open("W.txt", "w")
    for i in range(106):
        W_wirte.writelines(str(W[i]))
        W_wirte.write("\n")
    W_wirte.close()

    B_wirte = open("B.txt", "w")
    for i in range(1):
        B_wirte.write(str(B[i]))
    B_wirte.close()

    print("File:W / B / MAX / min write out Complete")

def test(file):
    W = np.zeros((106,))  # 權重
    B = np.zeros((1,))  # Bias
    MAX = np.zeros((106,))
    min = np.zeros((106,))



    #===========================================
    W_read = open("W.txt", "r")
    lines = W_read.readlines()
    A = 0
    for i in lines:
        W[A] = i
        A = A + 1
    W_read.close()
    #print(W[0])
    #===========================================

    #===========================================
    B_read = open("B.txt", "r")
    lines = B_read.readlines()
    B[0] = lines[0]
    B_read.close()
    #print(B[0])
    #===========================================

    # ===========================================
    MAX_read = open("MAX.txt", "r")
    lines = MAX_read.readlines()
    R = 0
    for line in lines:
        MAX[R] = line.strip('\n')
        R = R + 1
    #print(MAX)
    #print(MAX.shape)
    MAX_read.close()

    # ===========================================

    # ===========================================
    min_read = open("min.txt", "r")
    lines = min_read.readlines()
    R2 = 0
    for line in lines:
        min[R2] = line.strip('\n')
        R2 = R2 + 1
    #print(min)
    #print(min.shape)
    min_read.close()

    # ===========================================

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
    #print(X_test_Data_16281_106[0])


    X_test_Data_16281_106 = (X_test_Data_16281_106 - min) / (MAX - min)

    X_test.close()
    #print(X_test_Data_16281_106[0])

    #print(X_test_Data_16281_106[1])
    # =====================================================================

    #print(X_test_Data_16281_106.shape)
    #print(W.shape)
    #print(B.shape)

    #print(X_test_Data_16281_106[0])


    #Z = np.dot(float(X_test_Data_16281_106), float(W)) + float(B)
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

    print("1 counts : ",G)

#train()
#test("X_test.txt")


if "__main__" == __name__:
    '''''
    print('0 : ',sys.argv[0])
    print('1 : ',sys.argv[1])
    print('2 : ',sys.argv[2])
    print('3 : ',sys.argv[3])
    print('4 : ',sys.argv[4])
    '''''
    if sys.argv[2] == 'train':
        train()
    if sys.argv[2] == 'test':
        test(sys.argv[4])
