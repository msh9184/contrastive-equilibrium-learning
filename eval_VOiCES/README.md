# Contrastive equilibrium learning


## VOiCES Evaluation - unsupervised learning
We evaluate [VOiCES 2019 Challenge](https://iqtlabs.github.io/voices/) development set to verify the generalization of the out-of-domain data.


### Dataset and protocol for evaluation
The Voices Obscured in Complex Environmental Settings (VOiCES) corpus contains speech recorded by far-field microphones in noisy room conditions. VOiCES is publicly available [here](https://iqtlabs.github.io/voices/downloads/). We use the trial list on the development data provided in the VOiCES 2019 challenge.


### Evaluation
First, please download `dev-enroll.lst`, `dev-trial-keys.lst` and move them to a directory `./list`. And then make `trials_voices.txt` by running `python make_form.py`.
```bash
cd list
python make_form.py
```
Evaluation example on VOiCES. (In our setting)

```bash
python evaluate.py --initial_model ../save/a-cont/model/model000000001.model --save_path save/a-cont/ --save_filename model000000001 --test_list ./list/trials_voices.txt --test_path /home/shmun/DB/VOiCES/Development_Data/
```
