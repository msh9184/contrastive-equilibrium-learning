# Frist, run "bash makelist_path.sh" to create path_vox2.txt
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

# start loading
#spk = []
file = []
with open('path_vox2.txt') as f:
    lines = f.readlines()
    for i, v in enumerate(lines):
        tmp = v.strip().split('/')[8:]
        #spk.append(tmp[0])
        file.append(tmp[0]+'/'+tmp[1]+'/'+tmp[2])

#spk_to_num = generate_label(spk)
#print(len(np.unique(spk_to_num)))
print(len(file))

output = open('train_vox2.txt','w')
for i in range(len(file)):
    output.write(file[i]+'\n')
output.close()
