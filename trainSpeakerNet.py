#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys, time, os, argparse, socket
import numpy
import pdb
import torch
import glob
import zipfile
import datetime
from tuneThreshold import *
from SpeakerNet import SpeakerNet
from DatasetLoader import get_data_loader

parser = argparse.ArgumentParser(description = "SpeakerNet");

## Data loader
parser.add_argument('--max_frames', type=int, default=180,  help='Input length to the network');
parser.add_argument('--eval_frames', type=int, default=0,  help='Input length to the network');
parser.add_argument('--batch_size', type=int, default=200,  help='Batch size');
parser.add_argument('--nDataLoaderThread', type=int, default=20, help='Number of loader threads');

## Training details
parser.add_argument('--test_interval', type=int, default=5, help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int, default=500, help='Maximum number of epochs');
parser.add_argument('--unif_loss', type=str, default="uniform",    help='Uniformity loss function');
parser.add_argument('--sim_loss', type=str, default="anglecontrast",    help='Similarity loss function');
parser.add_argument('--augment_anchor', dest='augment_anchor', action='store_true', help='Augment anchor')
parser.add_argument('--augment_type',   type=int, default=3, help='0: no augment, 1: noise only, 2: noise or RIR, 3: noise and RIR');

## Learning rates
parser.add_argument('--lr', type=float, default=0.001,      help='Learning rate');
parser.add_argument("--lr_decay", type=float, default=0.95, help='Learning rate decay every [test_interval] epochs');
parser.add_argument("--lr_decay_interval", type=float, default=10, help='Learning rate decay interval');

## Load and save
parser.add_argument('--initial_model',  type=str, default="", help='Initial model weights');
parser.add_argument('--save_path',      type=str, default="./save/prac", help='Path for model and logs');

## Training and test data
parser.add_argument('--train_list', type=str, default="list/train_vox2.txt",   help='Train list');
parser.add_argument('--test_list',  type=str, default="list/test_vox1.txt",   help='Evaluation list');
parser.add_argument('--train_path', type=str, default="voxceleb2", help='Absolute path to the train set');
parser.add_argument('--test_path',  type=str, default="voxceleb1", help='Absolute path to the test set');
parser.add_argument('--musan_path',  type=str, default="musan_split", help='Absolute path to the test set');

## For test only
parser.add_argument('--eval', dest='eval', action='store_true', help='Eval only')
parser.add_argument('--trial_epoch', type=int, default=1,   help='Trial epoch number');

## Model definition
parser.add_argument('--model', type=str,        default="ResNetSE34L",     help='Name of model definition');
parser.add_argument('--encoder_type', type=str, default="SAP",  help='Type of encoder');
parser.add_argument('--nOut', type=int,         default=512,    help='Embedding size in the last FC layer');

## Hyperparameters
parser.add_argument('--lambda_s', type=float, default=1, help='Alignment weight');
parser.add_argument('--lambda_u', type=float, default=1, help='Uniformity weight');
parser.add_argument('--t', type=float, default=2, help='Uniformity parameter');
parser.add_argument('--sample_type', type=str, default="PoN", help='Uniformity parameter');

args = parser.parse_args();

## Initialise directories
model_save_path     = args.save_path+"/model"
result_save_path    = args.save_path+"/result"

if not(os.path.exists(model_save_path)):
    os.makedirs(model_save_path)
        
if not(os.path.exists(result_save_path)):
    os.makedirs(result_save_path)

## Load models
s = SpeakerNet(**vars(args));

it          = 1;
prevloss    = float("inf");
sumloss     = 0;

## Load model weights
modelfiles = glob.glob('%s/model0*.model'%model_save_path)
modelfiles.sort()

if len(modelfiles) >= 1:
    s.loadParameters(modelfiles[-1]);
    print("Model %s loaded from previous state!"%modelfiles[-1]);
    it = int(os.path.splitext(os.path.basename(modelfiles[-1]))[0][5:]) + 1
if(args.initial_model != ""):
    s.loadParameters(args.initial_model);
    print("Model %s loaded!"%args.initial_model);

for ii in range(0,it-1):
    if (ii!=0)  and (ii % args.lr_decay_interval == 0) :
        clr = s.updateLearningRate(args.lr_decay)

## Evaluation code
if args.eval == True:
        
    sc, lab, trials = s.evaluateFromList(args.test_list, print_interval=100, test_path=args.test_path, eval_frames=args.eval_frames)
    result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
    print('EER %2.4f'%result[1])

    quit();

## save code
pyfiles = glob.glob('./*.py')
strtime = datetime.datetime.now().strftime("%Y%m%d%H%M%S")

zipf = zipfile.ZipFile(result_save_path+ '/run%s.zip'%strtime, 'w', zipfile.ZIP_DEFLATED)
for file in pyfiles:
    zipf.write(file)
zipf.close()

f = open(result_save_path + '/run%s.cmd'%strtime, 'w')
f.write(' '.join(sys.argv))
f.close()

## Write args to scorefile
scorefile = open(result_save_path+"/scores.txt", "a+");

## Initialise data loader
trainLoader = get_data_loader(args.train_list, **vars(args));

clr = s.updateLearningRate(1)

while(1):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Training %s with LR %f..."%(args.model,max(clr)));

    ## Train network
    loss, traineer = s.train_network(loader=trainLoader);

    ## Validate and save
    if it % args.test_interval == 0:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), it, "Evaluating...");

        sc, lab, _ = s.evaluateFromList(args.test_list, print_interval=100, test_path=args.test_path, eval_frames=args.eval_frames)
        # eer
        result = tuneThresholdfromScore(sc, lab, [1, 0.1]);
        # dcf
        fnrs, fprs, thresholds = ComputeErrorRates(sc, lab)
        mindcf, threshold = ComputeMinDcf(fnrs, fprs, thresholds, p_target=0.05, c_miss=1, c_fa=1)

        print(args.save_path);
        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f, VDCF %2.4f"%( max(clr), traineer, loss, result[1], mindcf));
        scorefile.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f, VEER %2.4f VDCF %2.4f\n"%(it, max(clr), traineer, loss, result[1], mindcf));

        scorefile.flush()

        s.saveParameters(model_save_path+"/model%09d.model"%it);

        with open(model_save_path+"/model%09d.eer"%it, 'w') as eerfile:
            eerfile.write('%.4f'%result[1])

        with open(model_save_path+"/model%09d.dcf"%it, 'w') as eerfile:
            eerfile.write('%.4f'%mindcf)

    else:

        print(time.strftime("%Y-%m-%d %H:%M:%S"), "LR %f, TEER/TAcc %2.2f, TLOSS %f"%( max(clr), traineer, loss));
        scorefile.write("IT %d, LR %f, TEER/TAcc %2.2f, TLOSS %f\n"%(it, max(clr), traineer, loss));

        scorefile.flush()

    if it % args.lr_decay_interval == 0:
        clr = s.updateLearningRate(args.lr_decay)

    if it >= args.max_epoch:
        quit();

    it+=1;
    print("");

scorefile.close();





