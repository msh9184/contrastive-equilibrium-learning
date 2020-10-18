rm path_vox2.txt

find /home/shmun/DB/VoxCeleb/VoxCeleb2/dev/wav/ -name "*.wav" > tmp
sort tmp > path_vox2.txt
rm tmp

