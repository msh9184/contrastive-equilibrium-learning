import numpy as np

enr = np.loadtxt('dev-enroll.lst', str)
tst = np.loadtxt('dev-test.lst', str)
trial = np.loadtxt('dev-trial-keys.lst', str)

enr_p = []
gt = []
for i, (e, t, l) in enumerate(trial):
    idx = [k for k, x in enumerate(enr[:,0]) if x == e]
    enr_p.append(enr[idx[0]][1])
    if l == 'tgt':
        gt.append(1)
    else:
        gt.append(0)


with open('trials_voices.txt','w') as outfile:
    for i in range(len(gt)):
        outfile.write('%s %s %s\n'%(gt[i], enr_p[i], trial[i,1]))


