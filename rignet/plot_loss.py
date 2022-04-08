import os
import numpy as np 
import matplotlib.pyplot as plt


losstxt = '/home/uss00022/lelechen/github/StyleNeRF/rignet/checkpoints/rig/loss_log.txt'
reader = open(losstxt)
l = reader.readline()
l = reader.readline()
ss = 0

losses = {}
axis =[]
index = 0
while l:
        if index % 2 == 1:
                
                tmp = l[:-1].split(' ')[:-1]
                print (tmp)
                for i in range(0, len(tmp), 2):
                        print (tmp[i])
                        if tmp[i] not in losses.keys():
                                losses[tmp[i]] =[]
                        losses[tmp[i]].append(float(tmp[i+1]))

        index +=1
        l = reader.readline()
reader.close()
for k in losses.keys():
        axis = [i for i in range(len(losses[k]))]
        plt.plot(axis, losses[k], 'r--' )
        plt.show()
        plt.savefig('./gg/' + k[:-1] + '.png')