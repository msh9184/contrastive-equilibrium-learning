import numpy as np
import sys
import os

start_ep = int(sys.argv[-2])
end_ep = int(sys.argv[-1])

eer = []
eer_ep = []
for i in range(end_ep-start_ep+1):
    if os.path.exists("model%09d.eer"%(start_ep+i)) == True:
        tmp = np.loadtxt("model%09d.eer"%(start_ep+i))
    else:
        tmp = 999
    eer += [tmp]
    eer_ep += [start_ep+i]
eer_min = np.min(eer)
eer_min_ep = eer_ep[np.argmin(eer)]

dcf = []
dcf_ep = []
for i in range(end_ep-start_ep+1):
    if os.path.exists("model%09d.dcf"%(start_ep+i)) == True:
        tmp = np.loadtxt("model%09d.dcf"%(start_ep+i))
    else:
        tmp = 999
    dcf += [tmp]
    dcf_ep += [start_ep+i]
dcf_min = np.min(dcf)
dcf_min_ep = eer_ep[np.argmin(dcf)]

print(" ")
print("**********************************************************************")
print("*")
print("*  EER Min      :  "+str(eer_min))
print("*  EER Epoch    :  "+str(eer_min_ep))
print("*")
print("*  DCF Min      :  "+str(dcf_min))
print("*  DCF Epoch    :  "+str(dcf_min_ep))
print("*")
print("**********************************************************************")
print(" ")

