import os
import numpy as np 
import matplotlib.pyplot as plt


losstxt = '/home/uss00022/lelechen/github/CIPS-3D/checkpoints_debug2/Latent2Code/loss_log.txt'
reader = open(losstxt)
l = reader.readline()
l = reader.readline()
ss = 0
loss_land = []
loss_tex = []
axis =[]
while l:
    tmp = l[:-1].split(' ')
    l_land = tmp[7]
    l_tex =tmp[9]
    try:
        loss_land.append(float(l_land))
        loss_tex.append(float(l_tex))
    except:
        loss_land = []
        loss_tex = []
        
    # ss += 1
    # if ss == 1000:
    #     break
    l = reader.readline()
reader.close()
loss_tex = loss_tex[100:]
loss_land = loss_land[100:]

axis = [i for i in range(len(loss_tex))]
plt.plot(axis, loss_land, 'r--', axis, loss_tex, 'b--')
plt.show()
plt.savefig('./gg.png')