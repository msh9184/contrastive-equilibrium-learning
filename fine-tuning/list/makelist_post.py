# Frist, run "bash makelist_path.sh" to create path_xxxx.txt
# Then run "python makelist_post.py"

import numpy as np
import sys

def generate_label(label):
    label_unique = list(set(label))
    label_unique.sort()
    label_to_num = []
    for i, v in enumerate(label):
        label_to_num.append(label_unique.index(v))
    return np.array(label_to_num)

# VoxCeleb1 
spk1 = []
file1 = []
path1 = []
with open('path_vox1.txt') as f:
    lines = f.readlines()
    for i, v in enumerate(lines):
        tmp = v.strip().split('/')[8:]
        spk1.append(tmp[0])
        file1.append(tmp[0]+'/'+tmp[1]+'/'+tmp[2])
        path1.append(v.strip())

spk_to_num1 = generate_label(spk1)
print(len(np.unique(spk_to_num1)))
print(len(file1))

# VoxCeleb2
spk2 = []
file2 = []
path2 = []
with open('path_vox2.txt') as f:
    lines = f.readlines()
    for i, v in enumerate(lines):
        tmp = v.strip().split('/')[8:]
        spk2.append(tmp[0])
        file2.append(tmp[0]+'/'+tmp[1]+'/'+tmp[2])
        path2.append(v.strip())

spk_to_num2 = generate_label(spk2)
print(len(np.unique(spk_to_num2)))
print(len(file2))

# VoxCeleb1 + VoxCeleb2
spk12 = spk1 + spk2
path12 = path1 + path2

spk_to_num12 = generate_label(spk12)
print(len(np.unique(spk_to_num12)))
print(len(path12))


output1 = open('train_vox1.txt','w')
for i, (f, s) in enumerate(zip(file1, spk_to_num1)):
    output1.write(f+' '+str(s)+'\n')
output1.close()

output2 = open('train_vox2.txt','w')
for i, (f, s) in enumerate(zip(file2, spk_to_num2)):
    output2.write(f+' '+str(s)+'\n')
output2.close()

output12 = open('train_vox1-vox2.txt','w')
for i, (f, s) in enumerate(zip(path12, spk_to_num12)):
    output12.write(f+' '+str(s)+'\n')
output12.close()
