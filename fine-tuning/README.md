# Contrastive equilibrium learning

## Fine-tuning framwork
This repository provides fine-tuning implementation of [this paper](https://arxiv.org/abs/2010.11433).
This fine-tuning framework is modified on the [voxceleb_trainer](https://github.com/joonson/voxceleb_unsupervised).


### Dataset for fine-tuning
The [VoxCeleb](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/) datasets are used for these experiments.
The train list should contain the file path and the speaker identity, one line per utterance, as follows:
```bash
id10001/1zcIwhmdeo4/00001.wav 0
id10001/1zcIwhmdeo4/00002.wav 0
...
id11251/XHCSVYEZvlM/00001.wav 1210
id11251/XHCSVYEZvlM/00002.wav 1210
```
The train list for VoxCeleb1 can be download from [here](https://drive.google.com/file/d/174j4gCrdLBdo5sibBdNQi7WanA05WekM/view?usp=sharing) and the test list for VoxCeleb1 from [here](https://drive.google.com/file/d/1Lfb0bJAbE2zSCXfhLhJUQxDgro_mHRiq/view?usp=sharing).
The list (with labels) for training also can be created by runing `python makelist_post.py` in a directory `./list`.

In the fine-tuning experiments, data augmentation is not applied.

You can also follow the instructions on the following pages for download and the data preparation of training.
+ [Training](https://github.com/clovaai/voxceleb_trainer): VoxCeleb1&2 datasets


### Objective functions
```bash
Prototypical (proto)
A-Prototypical (angleproto)
A-Contrastive (anglecontrast)
GE2E (ge2e)
Softmax (softmax)
CosFace (amsoftmax)
ArcFace (aamsoftmax)
AdaCos (adasoftmax)
```

### Front-end encoders
```bash
FastResNet34 (ResNetSE34L)
VGGVox
TDNN
```

### Fine-tuning and evaluation.
Fine-tuning command line example with the development set of VoxCeleb1.

The fine-tuning example using `A-Contrast` loss function with the parameters pre-trained via CEL using `Uniformity` and `A-Prototypical` loss (i.e., `Unif + A-Prot -> A-Cont` as notation in [this paper](https://arxiv.org/abs/2010.11433)) as follows:
```bash
python fine-tuneSpeakerNet.py --initial_model ../save/pre-trained_a-prot.model --max_frames 300 --batch_size 250 --nSpeakers 1211 --trainfunc anglecontrast --save_path ./save/unif-a-prot_a-cont --train_list ./list/train_vox1.txt --test_list ./list/test_vox1.txt --train_path /home/shmun/DB/VoxCeleb/VoxCeleb1/dev/wav/ --test_path /home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/
```
where `/home/shmun/DB/VoxCeleb/VoxCeleb1/dev/wav/`, `/home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/` are our paths to VoxCeleb1 development set, VoxCeleb1 test set respectively. And `save/unif-a-prot_a-cont` is a directory to save results.

Evaluation command line example on the original test set of VoxCeleb1.
```bash
python fine-tuneSpeakerNet.py --eval --initial_model ./save/unif-a-prot_a-cont/model/model000000001.model --test_list ./list/test_vox1.txt --test_path /home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/
```


### Trained models
You can download the models reported in [this paper](https://arxiv.org/abs/2010.11433).

+  `VoxCeleb1` `Unif + A-Prot -> A-Prot` `EER: 2.33%`: [Download](https://drive.google.com/file/d/1TwCQ24KNVkNypgKg-1LF65oKqaINom3i/view?usp=sharing)
```bash
python fine-tuneSpeakerNet.py --eval --initial_model ./save/pre-trained_vox1_unif-a-prot_a-prot.model --test_list ./list/test_vox1.txt --test_path /home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/
```

+ `VoxCeleb2` `Unif + A-Cont -> ArcFace` `EER: 2.05%`: [Download](https://drive.google.com/file/d/1Pq9UW9h3sGv-Hwj_FcCKrkD8bjVHmXnn/view?usp=sharing)
```bash
python fine-tuneSpeakerNet.py --eval --initial_model ./save/pre-trained_vox2_unif-a-cont_arcface.model --test_list ./list/test_vox1.txt --test_path /home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/
```

+ `VoxCeleb1&2` `Unif + A-Cont -> GE2E` `EER: 1.81%`: [Download](https://drive.google.com/file/d/11Cyfb7do7sx7bycqpv1UDIj8SbbnDk47/view?usp=sharing)
```bash
python fine-tuneSpeakerNet.py --eval --initial_model ./save/pre-trained_vox1-vox2_unif-a-cont_ge2e.model --test_list ./list/test_vox1.txt --test_path /home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/
```
Code for VOiCES evaluation is [here](https://github.com/msh9184/contrastive-equilibrium-learning/tree/master/fine-tuning/eval_VOiCES).


### Citation
If you make use of this repository, please consider citing:
```
@article{mun2020cel,
  title={Unsupervised representation learning for speaker recognition via contrastive equilibrium learning},
  author={Mun, Sung Hwan and Kang, Woo Hyun and Han, Min Hyun and Kim, Nam Soo},
  journal={arXiv preprint arXiv:2010.11433},
  year={2020}
}
```