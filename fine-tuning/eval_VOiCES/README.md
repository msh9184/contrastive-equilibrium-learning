# Contrastive equilibrium learning


## VOiCES Evaluation - fine-tuning
We evaluate [VOiCES 2019 Challenge](https://iqtlabs.github.io/voices/) development set to verify the generalization of the out-of-domain data.


### Dataset and protocol for evaluation
The Voices Obscured in Complex Environmental Settings (VOiCES) corpus contains speech recorded by far-field microphones in noisy room conditions. VOiCES is publicly available [here](https://iqtlabs.github.io/voices/downloads/). We use the trial list on the development data provided in the VOiCES 2019 challenge.


### Evaluation
First, please download `dev-enroll.lst`, `dev-trial-keys.lst` from [here](https://iqtlabs.github.io/voices/downloads/) and move them to a directory `./list`. And then make `trials_voices.txt` by running `python make_form.py`.
```bash
cd list
python make_form.py
```
You can also download a `trials_voices.txt` file from [here](https://drive.google.com/file/d/1uCTIrDIl13hBDfQXYT4WITOlDYB-TgfW/view?usp=sharing) (672MB).

Evaluation example on VOiCES as follows:

```bash
python evaluate.py --initial_model ../save/unif-a-prot_a-cont/model/model000000001.model --save_path save/unif-a-prot_a-cont/ --save_filename model000000001 --test_list ./list/trials_voices.txt --test_path /home/shmun/DB/VOiCES/Development_Data/
```
where `/home/shmun/DB/VOiCES/Development_Data/` is our path to VOiCES development set and `save/unif-a-prot_a-cont` is a directory to save results.

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
