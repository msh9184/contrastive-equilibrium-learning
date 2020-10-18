rm path_vox1.txt

find /home/shmun/DB/VoxCeleb/VoxCeleb1/dev/wav/ -name "*.wav" > tmp
sort tmp > path_vox1.txt
rm tmp

rm path_vox2.txt

find /home/shmun/DB/VoxCeleb/VoxCeleb2/dev/wav/ -name "*.wav" > tmp
sort tmp > path_vox2.txt
rm tmp
