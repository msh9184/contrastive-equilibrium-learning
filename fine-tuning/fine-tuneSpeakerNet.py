#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import numpy
import pdb
import torch
import glob
from tuneThreshold import *
from SpeakerNet import SpeakerNet
from DatasetLoader import DatasetLoader

parser = argparse.ArgumentParser(description = "SpeakerNet");

## Data loader
parser.add_argument('--max_frames', type=int, default=300,  help='Input length to the network');
parser.add_argument('--eval_frames', type=int, default=0,  help='Input length to the network');
parser.add_argument('--batch_size', type=int, default=200,  help='Batch size');
parser.add_argument('--max_seg_per_spk', type=int, default=100, help='Maximum number of utterances per speaker per epoch');
parser.add_argument('--nDataLoaderThread', type=int, default=10, help='Number of loader threads');

## Training details
parser.add_argument('--test_interval', type=int, default=5, help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int, default=200, help='Maximum number of epochs');
parser.add_argument('--trainfunc', type=str, default="anglecontrast",    help='Loss function');
parser.add_argument('--trainfunc2', type=str, default="",    help='Loss function2');
parser.add_argument('--optimizer', type=str, default="adam", help='sgd or adam');

## Learning rates
parser.add_argument('--lr', type=float, default=0.001,      help='Learning rate');
parser.add_argument("--lr_decay", type=float, default=0.9, help='Learning rate decay every [lr_decay_interval] epochs');
parser.add_argument("--lr_decay_interval", type=float, default=10, help='Learning rate decay interval');

## Loss functions
parser.add_argument("--hard_prob", type=float, default=0, help='Hard negative mining probability, otherwise random, only for some loss functions');
parser.add_argument("--hard_rank", type=int, default=0,    help='Hard negative mining rank in the batch, only for some loss functions');
parser.add_argument('--margin', type=float,  default=0.2,     help='Loss margin, only for some loss functions');
parser.add_argument('--scale', type=float,   default=30,    help='Loss scale, only for some loss functions');
parser.add_argument('--nSpeakers', type=int, default=1211,  help='Number of speakers in the softmax layer for softmax-based losses');
parser.add_argument('--nPerSpeaker', type=int, default=2,  help='Number of utterances per speaker for metric-based losses');

## Load and save
parser.add_argument('--initial_model',  type=str, default="", help='Initial model weights');
parser.add_argument('--save_path',      type=str, default="./save/proc", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list', type=str, default="list/train_vox1.txt",   help='Train list');
parser.add_argument('--test_list',  type=str, default="list/test_vox1.txt",   help='Evaluation list');
parser.add_argument('--train_path', type=str, default="voxceleb1", help='Absolute path to the train set');
parser.add_argument('--test_path',  type=str, default="voxceleb1", help='Absolute path to the test set');

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')

## Model definition
parser.add_argument('--model', type=str,        default="ResNetSE34L", help='Name of model definition'); #ResNetSE34L
parser.add_argument('--encoder', type=str,      default="SAP",  help='Type of encoder');
parser.add_argument('--nOut', type=int,         default=512,    help='Embedding size in the last FC layer');

args = parser.parse_args();

# ==================== INITIALISE LINE NOTIFY ====================

if args.trainfunc2 == "uniform":
    uniform = True
else:
    uniform = False

## Initialise directories
model_save_path     = args.save_path+"/model"
result_save_path    = args.save_path+"/result"
feat_save_path      = ""

# ==================== MAKE DIRECTORIES ====================

if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)
        
if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)
else:
    print("Folder already exists. Press Enter to continue...")

# ==================== LOAD MODEL ====================

s = SpeakerNet(**vars(args));

# ==================== EVALUATE LIST ====================

it          = 1;
prevloss    = float("inf");
sumloss     = 0;
min_eer     = [];

# ==================== LOAD MODEL PARAMS ====================

modelfiles = glob.glob('%s/model0*.model'%model_save_path)
modelfiles.sort()

if len(modelfiles) >= 1:
    s.loadParameters(modelfiles[-1]);
    print("Model %s loaded from previous state!"%modelfiles[-1]);
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
elif(args.initial_model != ""):
    s.loadParameters(args.initial_model);
    print("Model %s loaded!"%args.initial_model);

for ii in range(1,it-1):
    if ii % args.lr_decay_interval == 0:
        clr = s.updateLearningRate(args.lr_decay) 

# ==================== EVAL ====================

if args.eval == True:
        
    sc, lab = s.evaluateFromListSave(listfilename=args.test_list, print_interval=100, feat_dir='', test_path=args.test_path, num_eval=10, eval_frames=args.eval_frames)
    result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
    print('EER %2.4f'%result[1])

    quit();

scorefile = open(result_save_path+"/scores.txt", "a+");

for items in vars(args):
    print(items, vars(args)[items]);
    scorefile.write('%s %s\n'%(items, vars(args)[items]));
scorefile.flush()

# ==================== ASSERTION ====================

gsize_dict  = {'triplet':2, 'contrastive':2, 'ge2e':args.nPerSpeaker, 'proto':args.nPerSpeaker, 'angleproto':args.nPerSpeaker, 'anglecontrast':args.nPerSpeaker, 'softmax':1, 'amsoftmax':1, 'aamsoftmax':1, 'adasoftmax':1}

assert args.trainfunc in gsize_dict
assert gsize_dict[args.trainfunc] <= 100

# ==================== CHECK SPK ====================

## print data stats
trainLoader = DatasetLoader(args.train_list, gSize=gsize_dict[args.trainfunc], **vars(args));

## update learning rate
clr = s.updateLearningRate(1)

while(1):   
    print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training %s with LR %.5f..."%(args.model,max(clr)));

    loss, traineer = s.train_network(loader=trainLoader);

    if it % args.test_interval == 0:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...");

        sc, lab = s.evaluateFromListSave(listfilename=args.test_list, print_interval=100, feat_dir='', test_path=args.test_path, num_eval=10, eval_frames=args.eval_frames)
        # eer
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
        # dcf
        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
        mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1)

        print(args.save_path);

        if uniform == True:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TAUC %2.2f, TLOSS %f, VEER %2.4f, VDCF %2.4f"%( max(clr), traineer, loss, result[1], mindcf));
            scorefile.write("IT %d, LR %f, TAcc %2.2f, TLOSS %f, VEER %2.4f VDCF %2.4f\n"%(it, max(clr), traineer, loss, result[1], mindcf));
        else:
            print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TAUC %2.2f, TLOSS %f, VEER %2.4f, VDCF %2.4f"%( max(clr), traineer, loss, result[1], mindcf));
            scorefile.write("IT %d, LR %f, TAUC %2.2f, TLOSS %f, VEER %2.4f VDCF %2.4f\n"%(it, max(clr), traineer, loss, result[1], mindcf));

        scorefile.flush()

        s.saveParameters(model_save_path+"/model%09d.model"%it);

        with open(model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
            eerfile.write('%.4f'%result[1])

        with open(model_save_path+"/model%09d.dcf"%it, 'w') as eerfile:
            eerfile.write('%.4f'%mindcf)

    else:
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TAUC %2.2f, TLOSS %f"%( max(clr), traineer, loss));
        scorefile.write("IT %d, LR %f, TAUC %2.2f, TLOSS %f\n"%(it, max(clr), traineer, loss));

        scorefile.flush()

    if it % args.lr_decay_interval == 0:
        clr = s.updateLearningRate(args.lr_decay)

    if it >= args.max_epoch:
        quit();

    it+=1;
    print("");

scorefile.close();

