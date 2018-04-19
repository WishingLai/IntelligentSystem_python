#####################################
#           A1035537_賴煒勛         #
#           2018/3/19 first_ver     #
#           2018/3/26 last_ver      #
#####################################

#import pandas as pd  # 引用套件並縮寫為 pd
import csv
import numpy as np
#import os
import sys
#import argparse
#from datetime import datetime
#from time import strftime

np.set_printoptions(threshold=np.inf)
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})  # 小數後2位

#ArrTraingData = np.array(TraingData)  # 轉 list 到 np_array
'''''
All_5652_163 = np.ones((5652, 163))  # 宣告 5652*163 的 matrix 最後一位是 b = 1
y_5652_1 = np.ones((5652, 1))        # 宣告 5652*1 的 matrix 存放pm2.5
y_pulun_5652_1 = np.ones((5652, 1))  # 宣告 5652*1 的 matrix 存放y'
OneMonth_18_480 = np.ones((18, 480)) # 宣告 18*480 的 matrix
ArrTD_B = 0                         #每一天其中一種測值的參數
ArrTD_C = 0                         #每 9小時 抓取 ArrTraingData 的位移量
EN_row = 0                          #大矩陣的行
D_A = 0                             #算 ArrTraingData 裡面的框框
D_B = 0                             #算 ArrTraingData 算框框裡的元素
P = 0
Shift_18_9 = 0                      #18*9 陣列的位移量
y_count = 0                         # y 的計算
W = np.ones((163, 1))               #加權值 w
L = np.zeros((5652, 1))
learing_rate = 1
gra = np.zeros((163, 1))
prev_gra = np.zeros((163, 1))
ada = np.zeros((163, 1))
'''''


def train():
    All_5652_163 = np.ones((5652, 163))  # 宣告 5652*163 的 matrix 最後一位是 b = 1
    y_5652_1 = np.ones((5652, 1))  # 宣告 5652*1 的 matrix 存放pm2.5
    y_pulun_5652_1 = np.ones((5652, 1))  # 宣告 5652*1 的 matrix 存放y'
    OneMonth_18_480 = np.ones((18, 480))  # 宣告 18*480 的 matrix
    ArrTD_B = 0  # 每一天其中一種測值的參數
    ArrTD_C = 0  # 每 9小時 抓取 ArrTraingData 的位移量
    EN_row = 0  # 大矩陣的行
    D_A = 0  # 算 ArrTraingData 裡面的框框
    D_B = 0  # 算 ArrTraingData 算框框裡的元素
    P = 0
    Shift_18_9 = 0  # 18*9 陣列的位移量
    y_count = 0  # y 的計算
    W = np.ones((163, 1))  # 加權值 w
    L = np.zeros((5652, 1))
    learing_rate = 1
    gra = np.zeros((163, 1))
    prev_gra = np.zeros((163, 1))
    ada = np.zeros((163, 1))
    with open('train.csv','r',newline="") as csvfile:
        reader = csv.reader(csvfile)

        R = 0

        #宣告TraingData list 抽取feature
        TraingData = []
        for rowCount in range(18):
            TraingData.append([])
        #print(TraingData)



        #抽取抽取feature
        deletFirst = 0
        for row in reader:
            if deletFirst == 0:
                deletFirst = 1
            else:
                TraingData[R % 18].append(row)
                R = R + 1

        #刪除日期、測站、測項
        for row in range(18):
            for col in range(20*12):
                del TraingData[row][col][0:3]


        CT = 0
        for TrainCount in TraingData:
            #print(CT)
            if CT == 10:                  #Replace NR with 0
                for row1 in range(20*12):
                    for col1 in range(24):
                        TrainCount[row1][col1] = TrainCount[row1][col1].replace("NR","0")




            #print(TrainCount)

            CT = CT +1


        ArrTraingData = np.array(TraingData)  # 轉 list 到 np_array
        '''''
        All_5652_163 = np.ones((5652, 163))  # 宣告 5652*163 的 matrix 最後一位是 b = 1
        y_5652_1 = np.ones((5652, 1))        # 宣告 5652*1 的 matrix 存放pm2.5
        y_pulun_5652_1 = np.ones((5652, 1))  # 宣告 5652*1 的 matrix 存放y'
        OneMonth_18_480 = np.ones((18, 480)) # 宣告 18*480 的 matrix
        ArrTD_B = 0                         #每一天其中一種測值的參數
        ArrTD_C = 0                         #每 9小時 抓取 ArrTraingData 的位移量
        EN_row = 0                          #大矩陣的行
        D_A = 0                             #算 ArrTraingData 裡面的框框
        D_B = 0                             #算 ArrTraingData 算框框裡的元素
        P = 0
        Shift_18_9 = 0                      #18*9 陣列的位移量
        y_count = 0                         # y 的計算
        W = np.ones((163, 1))               #加權值 w
        L = np.zeros((5652, 1))
        learing_rate = 1
        gra = np.zeros((163, 1))
        prev_gra = np.zeros((163, 1))
        ada = np.zeros((163, 1))
        '''''

        for BigMatrix in range(5652):
            #y  = wx + b

            #y_5652_1[BigMatrix][0] = ArrTraingData[][]

            #把每天製成 18*480 的一個月
            if (BigMatrix % 472) == 0 :     #什麼時候要在製作新的一個月
                #print("OneMonth_18_480 : " + str(BigMatrix))
                C_D_A = D_A
                for M_R in range(18):
                    for M_C in range(480):
                        if C_D_A >=240:
                            break
                        OneMonth_18_480[M_R][M_C] = ArrTraingData[M_R][C_D_A][D_B]

                        if M_R ==9 and M_C>8:
                            y_5652_1[y_count][0] = OneMonth_18_480[M_R][M_C]
    #                        print(y_5652_1[y_count])
                            y_count = y_count +1

                        if D_B == 23:
                            D_B = 0
                            C_D_A = C_D_A + 1
                        else:
                            D_B = D_B + 1
                    #print(M_R)
                    #print(OneMonth_18_480[M_R])
                    C_D_A = D_A                                     #C_D_A 加上框框位移量
                D_B = 0                                             #每個月跑完初始化
                D_A = D_A + 20                                      #算下個月的框框起始位置


    #       print("BIGMATRIX : " + str(BigMatrix))

            EigthteenNine = np.zeros((18, 9))       #宣告 18*9 的陣列: 每 9小時的 18測項

            for row2 in range(18):
                for col2 in range(9):
                    buf_col = Shift_18_9 + col2
                    #print(buf_col)
                    #print(Shift_18_9)
                    if buf_col >= 479:
                        Shift_18_9 = 0
                        break
                    else:
                        EigthteenNine[row2][col2] = OneMonth_18_480[row2][buf_col]  # ArrTraingData[][b][c]  =  b c 依情況改變

                np.set_printoptions(formatter={'float': '{: 0.2f}'.format})  # 小數後2位
                #print(EigthteenNine[row2])

            Shift_18_9 = Shift_18_9 + 1
            #print('=============================以上為[18*9]陣列=============================')



            A01 = 0
            B01 = 0
            for Count163 in range(162):
                if A01 == 18:
                    break
                All_5652_163[BigMatrix][Count163] = EigthteenNine[A01][B01]         #0-17 0-8
                B01 = B01 +1
                if B01 ==9:
                    A01 = A01 +1
                    B01 = 0

    #        print(All_5652_163[BigMatrix])
    #        print('=============================以上為 All_5652_163[5652*163]陣列==' + str(P) +'===========================')
            P = P+1


    #y_pulun_5652_1 = 5652_1
    #All_5652_163 = 5652_163
    #W = 163_1
    #L = 5652_1
    #gra = 163_1


        #print(type(All_5652_163))
        #print(type(W))

        i_th = 10000                      #training 次數
        learing_rate = 2
        for U in range(i_th):
            y_pulun_5652_1 = np.dot(All_5652_163, W)          #y’ = train_x 和 weight vector 的 內積
            L = y_pulun_5652_1 - y_5652_1               #L = y’ - train_y
            gra = 2*np.dot(All_5652_163.T, L)           #gra =
            prev_gra += gra**2
            ada = np.sqrt(prev_gra)
            W -= (learing_rate * gra) / ada

            #print(U,'  [W]: ', W)
            #print(U, '  [y_pulun_5652_1]: ', y_pulun_5652_1)
        N = 0
        for j in range(5652):                                   #判斷+-15差值
            if L[j]>15:
                #print('  [L]: ', L[j])
                N = N + 1
            if L[j]<-15:
                #print('  [L]: ', L[j])
                N = N + 1
            else:
                continue

        #print(N)


        with open('w.csv', 'w', newline="") as csv_file:  # 開 W加權值 _寫檔
            w_csv_writer = csv.writer(csv_file)
            for i in range(163):
                #print(W[i])
                w_csv_writer.writerow(W[i])
            csvfile.close()
    print('Training complete !')
    csvfile.close()


def test(file):
    with open(file,'r',newline="") as csvfile:
        smaple_reader = csv.reader(csvfile)

        #===============================================================================#=
        with open('w.csv', 'r', newline="") as csvfile: #讀取 w(加權檔案) 的值          #=
            w_reader = csv.reader(csvfile)                                              #=
            W = np.ones((163, 1))  # 加權值 w                                           #=
            Count11 = 0                                                                 #=
            for D in w_reader:                                                          #=
                W[Count11] = D                                                          #=
                Count11 = Count11 + 1                                                   #=
            #print(W)                                                                   #=
        csvfile.close()                                                                 #=
        # ================================================================================

        # 宣告TraingData list 抽取feature
        TraingData_sample = []
        for rowCount in range(18):                  #18項測值
            TraingData_sample.append([])
        # print(TraingData)

        R_sample = 0
        for row in smaple_reader:
            TraingData_sample[R_sample % 18].append(row)
            R_sample = R_sample + 1
        #R_sample = 198
        for row in range(18):
            for col in range(int(R_sample/18)):
                del TraingData_sample[row][col][0:2]

        #print(TraingData_sample[row])
        total = 9 * int(R_sample/18) #9*11 = 99
        #print(total)

        CT =0
        for TrainCount in TraingData_sample:
            # print(CT)
            if CT == 10:  # Replace NR with 0
                for row1 in range(int(R_sample/18)):
                    for col1 in range(9):
                        TrainCount[row1][col1] = TrainCount[row1][col1].replace("NR", "0")
            CT = CT +1
    #        print(TrainCount)

        OneMonth_18_total = np.ones((18, total))  # 宣告 18*480 的 matrix
        #print(OneMonth_18_total)
        R = 0
        C = 0

        ArrTraingData = np.array(TraingData_sample)
        #print(ArrTraingData)
        for row in range(18):
            for col in range(total):
                if R >= 11:
                    break
                OneMonth_18_total[row][col] = ArrTraingData[row][R][C]
                C = C + 1
                if C == 9:
                    R = R + 1
                    C = 0
                if R >= 11:
                    break
            R = 0
            C = 0
            #print(OneMonth_18_total[row])

        start = 0
        limit = 9
        csvfile.close()




    with open('predictions.csv', 'w', newline="") as csv_file:       #開_寫檔
        csv_writer = csv.writer(csv_file)
        head = (['id','value'])
        csv_writer.writerow(head)                        #寫檔上面欄位

        for V in range(int(R_sample/18)):
            One_162 = np.ones((1, 163))  # 宣告 1*162的陣列: 每 9小時的 18測項

            D = 0

            for B in range(18):
               for C in range(total):
                   C = C + start

                   if D>=162:
                       break
                   One_162[0][D] = OneMonth_18_total[B][C]
                   D = D + 1

                   if C+1 == limit:
                       C = 0
                       break

            start = start +9
            limit = limit +9

            y_pulun_1 = np.dot(One_162, W)

            print(y_pulun_1)
            PO = "%.2f" % y_pulun_1
            ROW = (['id_'+ str(V), float(PO)])
            csv_writer.writerow(ROW)
        csvfile.close()

if "__main__" == __name__:
    #print('0 : ',sys.argv[0])
    #print('1 : ',sys.argv[1])
    #print('2 : ',sys.argv[2])
    #print('3 : ',sys.argv[3])
    #print('4 : ',sys.argv[4])

    if sys.argv[2] == 'train':
        train()
    if sys.argv[2] == 'test':
        test(sys.argv[4])