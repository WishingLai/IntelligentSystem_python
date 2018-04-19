#  https://drive.google.com/file/d/10VBIBuWb9MQWMm9YFKjIzJjsNzzYFimy/view
#  https://drive.google.com/file/d/1b6C9YWaC2NGUOz2ca402_wU1c9Shqf_2/view

import csv
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import plotly.plotly as py
from pylab import figure, axes, pie, title, show
from random import shuffle

np.set_printoptions(threshold=np.nan)
np.set_printoptions(formatter={'float': '{: 0.6f}'.format})

Original_data_400_5 = np.ones((400, 5))
Train_data_400_5_Random = np.ones((400, 5))

with open('perceptron_train.dat', 'r', newline="") as file:
    RowReader = 0

    for row in file:  # 400
        Original_data_400_5[RowReader] = row.split()
        RowReader = RowReader + 1

    # 上面迴圈處理完原始的資料

    # print(Original_data_400_5[0])
    Complete_Flag_All_1_2000 = np.ones((1, 2000))

    for Out in range(2000):
       np.random.shuffle(Original_data_400_5)              # Random 資料集

       for In in range(400):                                # 把亂數排序後的資料放進 Train_data_400_5_Random
           Train_data_400_5_Random[In] = Original_data_400_5[In]

       #print('============================================================')
       #print('==============shuffle(Original_data_400_5) =================')
       #print('============================================================')

       # print(Train_data_400_5_Random[0])
       # print(Train_data_400_5_Random[0][4])

       i_th = 400  #次數
       Training_update = 0
       Complete_Flag = 0
       Z = 0
       Y = 0
       B = 0
       W_4_1 = np.zeros((4, 1))
       baios = 0
       U = 0
       #for U in range(i_th):
       while U < i_th:
           # 做一組(1,4)資料格式方便做運算
           Used_data = np.zeros((1, 4))
           for I in range(4):
               Used_data[0][I] = Train_data_400_5_Random[U][I]


           # print(Used_data)

           Z = np.dot(Used_data, W_4_1)                    # Z 等於預測值
           Y = Train_data_400_5_Random[U][4]               # Y 等於實際值
           P = Z + B

           if P * Y <= 0:      # (Z:- , Y:+) 或 (Z:+ , Y:-) 猜錯 需要調整 W,B
               #print(P * Y)
               W_4_1 = W_4_1 + Used_data.T * Train_data_400_5_Random[U][4]
               B = B + Train_data_400_5_Random[U][4]
               Training_update = Training_update + 1    #計算更新次數

           if P * Y > 0:        #預測正確
               Complete_Flag = Complete_Flag + 1


           if U == 399:
               if Complete_Flag == 400 :  # 跑了完整400次都沒有錯誤
                   #print('OK! Update how many times: ',Training_update)
                   Complete_Flag_All_1_2000[0][Out] = Training_update
                   break
               #print('同組再跑一次  ',U)
               #print('Complete_Flag  ', Complete_Flag)
               Complete_Flag = 0
               U = 0
               #print(U)
               continue

           U = U + 1

       #print(Complete_Flag_All_1_2000)


Count_num_2 = dict()

for A in range(2000):
    Count_num_2[Complete_Flag_All_1_2000[0][A]] = Count_num_2.get(Complete_Flag_All_1_2000[0][A], 0) + 1
    #book[word] = book.get(word, 0) + 1

#print(Count_num_2)


Times_Count = sorted(Count_num_2.items(), key=lambda d: d[1],reverse=True)
#print(Times_Count)
#print(len(Times_Count))

#plot
R = 100
R2 = 250
plt.title("Update_Count",color="r")
plt.xlabel("times",color="r")
plt.ylabel("count",color="r")
#in matplotlib,the output will be sorted by category, hence alphabetically...this is a bug


#plt.bar(range(len(Times_Count)), [Times_Count[i][1] for i in range(len(Times_Count))], align='center')
#plt.xticks(range(len(Times_Count)), [Times_Count[i][0] for i in range(len(Times_Count))])

plt.bar(list(Count_num_2.keys()),Count_num_2.values() , align='center')
#plt.xticks(range(len(Count_num_2), [Times_Count[i][0] for i in range(len(Times_Count))])

plt.savefig('hist.jpg')
plt.show()


print('Print Histogram!')