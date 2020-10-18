#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import numpy
import pdb
import torch
import glob
import zipfile
import datetime
#from tuneThreshold import tuneThresholdfromScore
from tuneThreshold import *
from SpeakerNet import SpeakerNet
from DatasetLoader import get_data_loader

parser = argparse.ArgumentParser(description = "SpeakerNet");

## Data loader
parser.add_argument('--eval_frames', type=int, default=0,  help='Input length to the network'); # 350
parser.add_argument('--nDataLoaderThread', type=int, default=20, help='Number of loader threads');

## Load and save
parser.add_argument('--initial_model',  type=str, default="", help='Initial model weights');
parser.add_argument('--save_path',  type=str, default="../save/exp", help='Save path');
parser.add_argument('--save_filename',  type=str, default="model000000000", help='Save filename');

## Training and test data
parser.add_argument('--test_list',  type=str, default="",   help='Evaluation list');
parser.add_argument('--test_path',  type=str, default="voxceleb1", help='Absolute path to the test set');

## Model definition
parser.add_argument('--model', type=str,        default="ResNetSE34L",     help='Name of model definition');
parser.add_argument('--encoder_type', type=str, default="SAP",  help='Type of encoder');
parser.add_argument('--nOut', type=int,         default=512,    help='Embedding size in the last FC layer');

args = parser.parse_args();

## Inference command
#CUDA_VISIBLE_DEVICES=0 python evaluate.py --initial_model ../../save/vox2_angleproto_uniform_fr180_bc200_aug3/model/model000000218.model --save_path save/model/ --save_filename 000000282

## Initialise args
args.test_path      = "/home/shmun/DB/VOiCES/Development_Data/"
args.test_list      = "list/trials_voices.txt"

## Load models
s = SpeakerNet(**vars(args));

if not(os.path.exists(args.save_path)):
    os.makedirs(args.save_path)

if(args.initial_model != ""):
    s.loadParameters(args.initial_model);
    print("Model %s loaded!"%args.initial_model);

## Evaluation code
# eer
sc, lab, trials = s.evaluateFromList(args.test_list, print_interval=10, test_path=args.test_path, eval_frames=args.eval_frames)
result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
# dcf
fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1)

print('EER %2.4f, MinDCF %2.4f'%(result[1], mindcf))

### Save scores
#with open(args.save_filename,'w') as outfile:
#    for vi, val in enumerate(sc):
#        outfile.write('%.4f %s\n'%(val,trials[vi]))

with open(args.save_path+args.save_filename+'.eer', 'w') as eerfile:
    eerfile.write('%.4f'%result[1])

with open(args.save_path+args.save_filename+'.dcf', 'w') as dcffile:
    dcffile.write('%.4f'%mindcf)


