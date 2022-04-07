import os
import numpy as np 
import matplotlib.pyplot as plt


losstxt = '/home/uss00022/lelechen/github/StyleNeRF/rignet/checkpoints/rig/loss_log.txt'
reader = open(losstxt)
l = reader.readline()
l = reader.readline()
ss = 0
loss_l2_v = []
loss_p_v = []
loss_land_v = []

loss_l2_w = []
loss_p_w = []
loss_land_w = []

axis =[]
while l:
        tmp = l[:-1].split(' ')
        l2_v = tmp[-4]
        p_v =tmp[-2]
        land_v = tmp[-6]
        p_v =tmp[-2]

        l2_w = tmp[-10]
        p_w =tmp[-8]
        land_v = tmp[-12]


        loss_l2_v.append(float(l2_v))
        loss_l2_w.append(float(l2_w))

        loss_p_v.append(float(p_v))
        loss_p_w.append(float(p_w))

        loss_land_w.append(float(land_w))
        loss_land_v.append(float(land_v))
    
        l = reader.readline()
reader.close()

# loss_tex = loss_tex[100:]
# loss_land = loss_land[100:]

axis = [i for i in range(len(loss_l2_v))]
print (axis)
plt.plot(axis, loss_l2_v, 'r--', axis, loss_p_v, 'b--',axis, loss_land_v, 'g--', axis, loss_l2_w, 'r*', axis, loss_p_w, 'b*',axis, loss_land_w, 'g*')
plt.show()
plt.savefig('./gg.png')