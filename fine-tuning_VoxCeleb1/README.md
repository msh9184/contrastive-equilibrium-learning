# Contrastive equilibrium learning


## Fine-tuning framwork
This fine-tuning framework is modified on the [voxceleb_trainer](https://github.com/joonson/voxceleb_unsupervised) provided by [this paper]().


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
The list (with labels) for training also can be created by runing `python makelist_post.py` in a directory `./make_list`.

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
Fine-tuning command line with the development set of VoxCeleb1.
The fine-tuning example using `A-Contrast` loss function and the CEL model pre-trained with `A-Prototypical` and `Uniformity` loss (i.e., `A-Prot + Unif ¡æ A-Cont` in a [paper]()) as follows:
```bash
python fine-tuneSpeakerNet.py --initial_model ../save/pre-trained_a-prot.model --max_frames 300 --batch_size 250 --nSpeakers 1211 --trainfunc anglecontrast --save_path ./save/a-prot-unif_a-cont --train_list ./list/train_vox1.txt --test_list ./list/test_vox1.txt --train_path /home/shmun/DB/VoxCeleb/VoxCeleb1/dev/wav/ --test_path /home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/
```
where `/home/shmun/DB/VoxCeleb/VoxCeleb1/dev/wav/`, `/home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/` are the paths to VoxCeleb1 development set, VoxCeleb1 test set respectively. And `save/a-prot-unif_a-cont` is a directory to save results.

Evaluation command line on the original test set of VoxCeleb1.
```bash
python fine-tuneSpeakerNet.py --eval --initial_model ./save/a-prot-unif_a-cont/model/model000000001.model --test_list ./list/test_vox1.txt --test_path /home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/
```


### Trained models
You can download the models reported in [this paper]().

+  `A-Prot + Unif ¡æ A-Prot` `VoxCeleb1` `EER: 2.33%`: [Download]()
```bash
python trainSpeakerNet.py --eval --initial_model ./save/pre-trained_a-prot-unif_a-prot.model --test_list ./list/test_vox1.txt --test_path /home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/
```

+ `A-Cont + Unif ¡æ CosFace` `VoxCeleb2` `EER: 2.07%`: [Download]()
```bash
python trainSpeakerNet.py --eval --initial_model ./save/pre-trained_a-cont-unif_cosface.model --test_list ./list/test_vox1.txt --test_path /home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/
```

+ `A-Cont + Unif ¡æ A-Cont` `VoxCeleb1&2` `EER: 1.88%`: [Download]()
```bash
python trainSpeakerNet.py --eval --initial_model ./save/pre-trained_a-cont-unif_a-cont.model --test_list ./list/test_vox1.txt --test_path /home/shmun/DB/VoxCeleb/VoxCeleb1/test/wav/
```
Code for VOiCES evaluation is [here]().
