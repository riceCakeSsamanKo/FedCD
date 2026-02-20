#!/usr/bin/env python
import copy
import torch
import argparse
import os
import time
import json
import warnings
import re
import numpy as np
import torchvision
import logging
import sys

# Allow running from repo root (e.g., python .\system\main.py)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from flcore.servers.serverfedcd import FedCD # FedCD
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverpFedMe import pFedMe
from flcore.servers.serverperavg import PerAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverfomo import FedFomo
from flcore.servers.serveramp import FedAMP
from flcore.servers.servermtl import FedMTL
from flcore.servers.serverlocal import Local
from flcore.servers.serverper import FedPer
from flcore.servers.serverapfl import APFL
from flcore.servers.serverditto import Ditto
from flcore.servers.serverrep import FedRep
from flcore.servers.serverphp import FedPHP
from flcore.servers.serverbn import FedBN
from flcore.servers.serverrod import FedROD
from flcore.servers.serverproto import FedProto
from flcore.servers.serverdyn import FedDyn
from flcore.servers.servermoon import MOON
from flcore.servers.serverbabu import FedBABU
from flcore.servers.serverapple import APPLE
from flcore.servers.servergen import FedGen
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.servers.serverfd import FD
from flcore.servers.serverala import FedALA
from flcore.servers.serverlg import LG_FedAvg
from flcore.servers.servergc import FedGC
from flcore.servers.serverfml import FML
from flcore.servers.serverkd import FedKD
from flcore.servers.serverpcl import FedPCL
from flcore.servers.servercp import FedCP
from flcore.servers.servergpfl import GPFL
from flcore.servers.serverntd import FedNTD
from flcore.servers.servergh import FedGH
from flcore.servers.serverdbe import FedDBE
from flcore.servers.servercac import FedCAC
from flcore.servers.serverda import PFL_DA
from flcore.servers.serverlc import FedLC
from flcore.servers.serveras import FedAS
from flcore.servers.servercross import FedCross

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")
torch.manual_seed(0)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    def build_model(model_name):
        if model_name == "MLR": # convex
            if "MNIST" in args.dataset:
                return Mclr_Logistic(1*28*28, num_classes=args.num_classes).to(args.device)
            if "Cifar10" in args.dataset:
                return Mclr_Logistic(3*32*32, num_classes=args.num_classes).to(args.device)
            return Mclr_Logistic(60, num_classes=args.num_classes).to(args.device)

        if model_name == "CNN": # non-convex
            if "MNIST" in args.dataset:
                return FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024).to(args.device)
            if "Cifar10" in args.dataset:
                return FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600).to(args.device)
            if "Omniglot" in args.dataset:
                return FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856).to(args.device)
            if "Digit5" in args.dataset:
                return Digit5CNN().to(args.device)
            return FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816).to(args.device)

        if model_name == "DNN": # non-convex
            if "MNIST" in args.dataset:
                return DNN(1*28*28, 100, num_classes=args.num_classes).to(args.device)
            if "Cifar10" in args.dataset:
                return DNN(3*32*32, 100, num_classes=args.num_classes).to(args.device)
            return DNN(60, 20, num_classes=args.num_classes).to(args.device)
        
        if model_name == "ResNet18":
            return torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes).to(args.device)
        
        if model_name == "ResNet10":
            return resnet10(num_classes=args.num_classes).to(args.device)
        
        if model_name == "ResNet34":
            return torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes).to(args.device)

        if model_name == "VGG16":
            return torchvision.models.vgg16(pretrained=False, num_classes=args.num_classes).to(args.device)

        if model_name == "VGG8":
            return VGG8(num_classes=args.num_classes).to(args.device)

        if model_name.startswith("VGG8W"):
            match = re.fullmatch(r"VGG8W(\d+)", model_name)
            if match is None:
                raise ValueError(
                    f"Invalid model name: {model_name}. "
                    "Use VGG8 or VGG8W<hidden>, e.g., VGG8W768."
                )
            hidden_dim = int(match.group(1))
            if hidden_dim <= 0:
                raise ValueError(f"Invalid VGG8 hidden width: {hidden_dim}")
            return VGG8(num_classes=args.num_classes, classifier_hidden=hidden_dim).to(args.device)

        if model_name == "AlexNet":
            return alexnet(pretrained=False, num_classes=args.num_classes).to(args.device)
        
        if model_name == "GoogleNet":
            return torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=args.num_classes).to(args.device)

        if model_name == "MobileNet":
            return mobilenet_v2(pretrained=False, num_classes=args.num_classes).to(args.device)
            
        if model_name == "LSTM":
            return LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        if model_name == "BiLSTM":
            return BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, 
                                             output_size=args.num_classes, num_layers=1, 
                                             embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                                             embedding_length=args.feature_dim).to(args.device)

        if model_name == "fastText":
            return fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes).to(args.device)

        if model_name == "TextCNN":
            return TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                           num_classes=args.num_classes).to(args.device)

        if model_name == "Transformer":
            return TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, 
                                    num_classes=args.num_classes, max_len=args.max_len).to(args.device)
        
        if model_name == "AmazonMLP":
            return AmazonMLP().to(args.device)

        if model_name == "HARCNN":
            if args.dataset == 'HAR':
                return HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                              pool_kernel_size=(1, 2)).to(args.device)
            if args.dataset == 'PAMAP2':
                return HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                              pool_kernel_size=(1, 2)).to(args.device)

        raise NotImplementedError

    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        args.model = build_model(model_str)
        if args.algorithm == "FedCD":
            pm_name = getattr(args, "pm_model_name", None)
            if pm_name:
                args.pm_model = build_model(pm_name)

        print(args.model)

        # FedCD: keep base model on CPU to avoid GPU OOM
        if args.algorithm == "FedCD" and args.device == "cuda" and args.avoid_oom == True:
            args.model = args.model.to("cpu")
            if getattr(args, "pm_model", None) is not None:
                args.pm_model = args.pm_model.to("cpu")

        # select algorithm
        if args.algorithm == "FedCD":
            server = FedCD(args, i) # FedCD 추가
        
        elif args.algorithm == "FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAvg(args, i)

        elif args.algorithm == "Local":
            server = Local(args, i)

        elif args.algorithm == "FedMTL":
            server = FedMTL(args, i)

        elif args.algorithm == "PerAvg":
            server = PerAvg(args, i)

        elif args.algorithm == "pFedMe":
            server = pFedMe(args, i)

        elif args.algorithm == "FedProx":
            server = FedProx(args, i)

        elif args.algorithm == "FedFomo":
            server = FedFomo(args, i)

        elif args.algorithm == "FedAMP":
            server = FedAMP(args, i)

        elif args.algorithm == "APFL":
            server = APFL(args, i)

        elif args.algorithm == "FedPer":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "FedPAC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            from flcore.servers.serverpac import FedPAC
            server = FedPAC(args, i)

        elif args.algorithm == "LG-FedAvg":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FedGC":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGC(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedKD(args, i)

        elif args.algorithm == "FedPCL":
            args.model.fc = nn.Identity()
            server = FedPCL(args, i)

        elif args.algorithm == "FedCP":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedCP(args, i)

        elif args.algorithm == "GPFL":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = GPFL(args, i)

        elif args.algorithm == "FedNTD":
            server = FedNTD(args, i)

        elif args.algorithm == "FedGH":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedGH(args, i)

        elif args.algorithm == "FedDBE":
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedDBE(args, i)

        elif args.algorithm == 'FedCAC':
            server = FedCAC(args, i)

        elif args.algorithm == 'PFL-DA':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = PFL_DA(args, i)

        elif args.algorithm == 'FedLC':
            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedLC(args, i)

        elif args.algorithm == 'FedAS':

            args.head = copy.deepcopy(args.model.fc)
            args.model.fc = nn.Identity()
            args.model = BaseHeadSplit(args.model, args.head)
            server = FedAS(args, i)
            
        elif args.algorithm == "FedCross":
            server = FedCross(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    if args.log_usage_path:
        result_dir = os.path.dirname(args.log_usage_path)
    else:
        result_dir = "../results/"
        
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times, result_path=result_dir)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="train", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="VGG16")
    parser.add_argument('-lbs', "--batch_size", type=int, default=128)
    parser.add_argument('-nw', "--num_workers", type=int, default=0,
                        help="DataLoader workers; >0 can increase GPU utilization")
    parser.add_argument('--pin_memory', type=str2bool, default=True,
                        help="Pin host memory for faster GPU transfer")
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help="DataLoader prefetch factor (num_workers>0)")
    parser.add_argument('--gpu_batch_mult', type=int, default=1,
                        help="Multiply batch size on GPU (FedCD safe scaling)")
    parser.add_argument('--gpu_batch_max', type=int, default=0,
                        help="Max GPU batch size (0 = no cap)")
    parser.add_argument('--amp', type=str2bool, default=True,
                        help="Use mixed precision on CUDA for speed")
    parser.add_argument('--tf32', type=str2bool, default=True,
                        help="Enable TF32 on CUDA for speed")
    parser.add_argument('--log_usage', type=str2bool, default=False,
                        help="Log CPU/GPU usage each round")
    parser.add_argument('--log_usage_every', type=int, default=1,
                        help="Log usage every N rounds")
    parser.add_argument('--log_usage_path', type=str, default="logs/usage.csv",
                        help="Usage log CSV path")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=str2bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedCD")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=str2bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=10,
                        help="Total number of clients")
    # FedCD
    parser.add_argument('--num_clusters', type=int, default=5)
    parser.add_argument('--cluster_threshold', type=float, default=0.0,
                        help="Initial distance threshold for dynamic clustering. If > 0, num_clusters is ignored.")
    parser.add_argument('--adaptive_threshold', type=str2bool, default=False,
                        help="Enable adaptive threshold adjustment based on client performance trends.")
    parser.add_argument('--threshold_step', type=float, default=0.01,
                        help="Step size for increasing/decreasing the clustering threshold (used if rates are not specified).")
    parser.add_argument('--threshold_step_max', type=float, default=0.1,
                        help="Maximum absolute threshold change per ACT update.")
    parser.add_argument('--threshold_decay', type=float, default=0.9,
                        help="Decay rate for the threshold step size when direction reverses (Zig-Zag).")
    parser.add_argument('--act_window_size', type=int, default=5,
                        help="Sliding window size for regression-based adaptive threshold (ACT).")
    parser.add_argument('--act_min_slope', type=float, default=0.0002,
                        help="Minimum slope threshold to consider performance as 'improving' in ACT.")
    parser.add_argument('--threshold_inc_rate', type=float, default=1.3,
                        help="Multiplier for increasing the clustering threshold (e.g., 1.3 for 30%% increase).")
    parser.add_argument('--threshold_dec_rate', type=float, default=0.5,
                        help="Multiplier for decreasing the clustering threshold (e.g., 0.5 for 50%% decrease).")
    parser.add_argument('--threshold_max', type=float, default=0.95,
                        help="Maximum limit for the clustering threshold.")
    parser.add_argument('--ema_alpha', type=float, default=0.3,
                        help="Exponential Moving Average alpha for client performance trend.")
    parser.add_argument('--tolerance_ratio', type=float, default=0.4,
                        help="Ratio of clients allowed to degrade before shrinking clusters.")
    parser.add_argument('--cluster_period', type=int, default=2)
    parser.add_argument('--pm_period', type=int, default=1,
                        help="PM aggregation/broadcast period (global rounds)")
    parser.add_argument('--global_period', type=int, default=4)
    parser.add_argument('--fedcd_enable_clustering', type=str2bool, default=True,
                        help="Enable periodic client clustering by distribution similarity.")
    parser.add_argument('--fedcd_enable_pm_aggregation', type=str2bool, default=True,
                        help="Enable cluster-wise PM aggregation and PM broadcast to clients.")
    parser.add_argument('--cluster_sample_size', type=int, default=512)
    parser.add_argument('--max_dynamic_clusters', type=int, default=5,
                        help="Maximum number of clusters allowed in threshold-based dynamic clustering (0 disables cap).")
    parser.add_argument('--fedcd_nc_weight', type=float, default=0.0)
    parser.add_argument('--fedcd_nc_target_corr', type=float, default=-0.1,
                        help="Target feature-wise correlation between GM/PM features for NC regularization.")
    parser.add_argument('--fedcd_fusion_weight', type=float, default=1.0,
                        help="Main CE/NLL loss weight for entropy-mixed GM+PM prediction.")
    parser.add_argument('--fedcd_pm_logits_weight', type=float, default=0.5,
                        help="Auxiliary CE loss weight for PM logits branch during local training.")
    parser.add_argument('--fedcd_pm_only_weight', type=float, default=1.5,
                        help="Extra PM-only CE loss weight during local training.")
    parser.add_argument('--fedcd_gm_logits_weight', type=float, default=1.0,
                        help="Auxiliary CE loss weight for GM-only logits during local training.")
    parser.add_argument('--fedcd_local_pm_only_objective', type=str2bool, default=False,
                        help="If True, local training optimizes only PM-only CE (ignores fusion/GM losses).")
    parser.add_argument('--fedcd_gm_lr_scale', type=float, default=0.1,
                        help="Local GM learning-rate scale relative to base local lr.")
    parser.add_argument('--fedcd_gm_update_mode', type=str, default="local",
                        help="GM update mode: local | server_pm_teacher | server_pm_fedavg | server_proto_teacher | hybrid_local_proto")
    parser.add_argument('--fedcd_hybrid_proto_blend', type=float, default=0.35,
                        help="Blend ratio for hybrid_local_proto mode (0=local FedAvg only, 1=prototype-only).")
    parser.add_argument('--fedcd_entropy_temp_pm', type=float, default=1.0,
                        help="Temperature for PM probabilities in entropy-based PM/GM mixing.")
    parser.add_argument('--fedcd_entropy_temp_gm', type=float, default=1.0,
                        help="Temperature for GM probabilities in entropy-based PM/GM mixing.")
    parser.add_argument('--fedcd_entropy_min_pm_weight', type=float, default=0.1,
                        help="Minimum PM mixing weight in entropy gate.")
    parser.add_argument('--fedcd_entropy_max_pm_weight', type=float, default=0.9,
                        help="Maximum PM mixing weight in entropy gate.")
    parser.add_argument('--fedcd_entropy_gate_tau', type=float, default=0.2,
                        help="Temperature for PM-vs-GM confidence gate (smaller = harder switching).")
    parser.add_argument('--fedcd_entropy_pm_bias', type=float, default=0.0,
                        help="Additive PM confidence bias in entropy gate.")
    parser.add_argument('--fedcd_entropy_gm_bias', type=float, default=0.0,
                        help="Additive GM confidence bias in entropy gate.")
    parser.add_argument('--fedcd_entropy_disagree_gm_boost', type=float, default=0.0,
                        help="Extra PM-weight reduction when PM/GM disagree and GM confidence is higher.")
    parser.add_argument('--fedcd_entropy_use_class_reliability', type=str2bool, default=True,
                        help="Use per-class PM/GM reliability (EMA) to modulate entropy gate.")
    parser.add_argument('--fedcd_entropy_reliability_scale', type=float, default=0.7,
                        help="Strength of per-class reliability modulation in entropy gate.")
    parser.add_argument('--fedcd_entropy_hard_switch_margin', type=float, default=0.15,
                        help="Confidence-gap margin for hard PM/GM branch selection (0 disables).")
    parser.add_argument('--fedcd_entropy_use_ood_gate', type=str2bool, default=True,
                        help="Use local feature-distribution similarity (OOD-aware) to suppress PM on OOD samples.")
    parser.add_argument('--fedcd_entropy_ood_scale', type=float, default=1.0,
                        help="Scale of OOD-aware PM suppression; smaller means stronger suppression.")
    parser.add_argument('--fedcd_fusion_mode', type=str, default="soft",
                        help="Inference fusion mode: soft | pm_defer_hard.")
    parser.add_argument('--fedcd_pm_defer_conf_threshold', type=float, default=0.55,
                        help="For pm_defer_hard, minimum PM confidence (1-normalized entropy) to keep PM.")
    parser.add_argument('--fedcd_pm_defer_gm_margin', type=float, default=0.02,
                        help="For pm_defer_hard, switch to GM if GM confidence exceeds PM by this margin.")
    parser.add_argument('--fedcd_pm_defer_ood_threshold', type=float, default=0.35,
                        help="For pm_defer_hard, minimum in-distribution score to keep PM when OOD gate is enabled.")
    parser.add_argument('--fedcd_gate_reliability_ema', type=float, default=0.9,
                        help="EMA factor for updating per-class PM/GM gate reliability.")
    parser.add_argument('--fedcd_gate_reliability_samples', type=int, default=512,
                        help="Max local test samples per round for gate reliability estimation (0 = full test set).")
    parser.add_argument('--fedcd_gate_feature_ema', type=float, default=0.9,
                        help="EMA factor for updating local feature-distribution stats used by OOD gate.")
    parser.add_argument('--fedcd_gate_feature_samples', type=int, default=512,
                        help="Max local samples per round for feature-stat updates (0 = full local train set).")
    parser.add_argument('--fedcd_warmup_epochs', type=int, default=0)
    parser.add_argument('--fedcd_pm_teacher_lr', type=float, default=0.01,
                        help="Server PM-teacher distillation learning rate for GM update.")
    parser.add_argument('--fedcd_pm_teacher_temp', type=float, default=2.0,
                        help="Temperature for server PM-teacher distillation.")
    parser.add_argument('--fedcd_pm_teacher_kl_weight', type=float, default=1.0,
                        help="KL loss weight for server PM-teacher distillation.")
    parser.add_argument('--fedcd_pm_teacher_ce_weight', type=float, default=0.2,
                        help="CE loss weight (with labels) for server PM-teacher distillation.")
    parser.add_argument('--fedcd_pm_teacher_samples', type=int, default=2000,
                        help="Number of samples for server PM-teacher distillation (0 = full proxy dataset).")
    parser.add_argument('--fedcd_pm_teacher_batch_size', type=int, default=256,
                        help="Batch size for server PM-teacher distillation.")
    parser.add_argument('--fedcd_pm_teacher_epochs', type=int, default=1,
                        help="Number of epochs over PM-teacher distillation dataset per GM update.")
    parser.add_argument('--fedcd_pm_teacher_proxy_dataset', type=str, default="Cifar100",
                        help="Proxy dataset for PM-teacher distillation (e.g., Cifar100, TinyImagenet).")
    parser.add_argument('--fedcd_pm_teacher_proxy_root', type=str, default="",
                        help="Root path for proxy dataset. Empty uses built-in default path.")
    parser.add_argument('--fedcd_pm_teacher_proxy_split', type=str, default="train",
                        help="Proxy split: train | test | all.")
    parser.add_argument('--fedcd_pm_teacher_proxy_download', type=str2bool, default=False,
                        help="Download proxy dataset when supported (CIFAR10/100).")
    parser.add_argument('--fedcd_pm_teacher_allow_test_fallback', type=str2bool, default=False,
                        help="Allow fallback to target test-union distill set if proxy cannot be loaded.")
    parser.add_argument('--fedcd_pm_teacher_source', type=str, default="cluster",
                        help="Teacher source for server_pm_teacher GM distillation: cluster | client.")
    parser.add_argument('--fedcd_pm_teacher_confidence_weight', type=str2bool, default=True,
                        help="Enable teacher-confidence weighting for KL distillation.")
    parser.add_argument('--fedcd_pm_teacher_confidence_min', type=float, default=0.05,
                        help="Minimum per-sample KL weight when confidence weighting is enabled.")
    parser.add_argument('--fedcd_pm_teacher_confidence_power', type=float, default=1.0,
                        help="Exponent for teacher confidence shaping in KL weighting.")
    parser.add_argument('--fedcd_pm_teacher_ensemble_confidence', type=str2bool, default=True,
                        help="Use per-sample teacher confidence when ensembling multiple PM teachers.")
    parser.add_argument('--fedcd_pm_teacher_topk', type=int, default=0,
                        help="Top-k PM teachers per sample for GM distillation (0 = use all teachers).")
    parser.add_argument('--fedcd_pm_teacher_abstain_threshold', type=float, default=0.0,
                        help="Skip distillation for samples whose best teacher confidence is below this threshold (0~1).")
    parser.add_argument('--fedcd_pm_teacher_teacher_abstain_threshold', type=float, default=0.0,
                        help="Drop individual PM teachers for a sample when teacher confidence is below this threshold (0~1).")
    parser.add_argument('--fedcd_pm_teacher_min_active_teachers', type=int, default=1,
                        help="Minimum number of active PM teachers required per sample after filtering.")
    parser.add_argument('--fedcd_pm_teacher_consensus_min_ratio', type=float, default=0.0,
                        help="Minimum weighted teacher majority ratio per sample for distillation (0~1, 0 disables).")
    parser.add_argument('--fedcd_pm_teacher_correct_only', type=str2bool, default=False,
                        help="Distill only samples whose PM-teacher ensemble top-1 matches label (requires label-compatible proxy dataset).")
    parser.add_argument('--fedcd_pm_teacher_rel_weight', type=float, default=0.2,
                        help="Relational KD weight (sample-similarity matching) for PM-teacher GM distillation.")
    parser.add_argument('--fedcd_pm_teacher_rel_batch', type=int, default=64,
                        help="Max batch size used for relational KD similarity matching.")
    parser.add_argument('--fedcd_init_pretrain', type=str2bool, default=True,
                        help="Run server-side proxy initialization pretrain for f_ext/GM/PM before round 0.")
    parser.add_argument('--fedcd_init_epochs', type=int, default=1,
                        help="Epochs for server-side initialization pretrain.")
    parser.add_argument('--fedcd_init_lr', type=float, default=0.005,
                        help="Learning rate for server-side initialization pretrain.")
    parser.add_argument('--fedcd_init_samples', type=int, default=2000,
                        help="Proxy samples used in server-side initialization pretrain (0 = full proxy set).")
    parser.add_argument('--fedcd_init_batch_size', type=int, default=256,
                        help="Batch size for server-side initialization pretrain.")
    parser.add_argument('--fedcd_init_ce_weight', type=float, default=1.0,
                        help="CE weight in initialization pretrain (applied when proxy labels match target classes).")
    parser.add_argument('--fedcd_init_kd_weight', type=float, default=1.0,
                        help="Mutual KD weight between GM and PM during initialization pretrain.")
    parser.add_argument('--fedcd_init_entropy_weight', type=float, default=0.05,
                        help="Entropy regularization weight during initialization pretrain.")
    parser.add_argument('--fedcd_init_diversity_weight', type=float, default=0.05,
                        help="Batch diversity regularization weight during initialization pretrain.")
    parser.add_argument('--fedcd_proto_teacher_lr', type=float, default=0.01,
                        help="Server prototype-teacher learning rate for GM update.")
    parser.add_argument('--fedcd_proto_teacher_steps', type=int, default=200,
                        help="Optimization steps for prototype-based server GM update.")
    parser.add_argument('--fedcd_proto_teacher_batch_size', type=int, default=256,
                        help="Batch size for synthetic prototype sampling during GM update.")
    parser.add_argument('--fedcd_proto_teacher_temp', type=float, default=2.0,
                        help="Temperature for prototype-teacher KL loss.")
    parser.add_argument('--fedcd_proto_teacher_ce_weight', type=float, default=1.0,
                        help="CE loss weight (pseudo labels from class prototypes).")
    parser.add_argument('--fedcd_proto_teacher_kl_weight', type=float, default=0.5,
                        help="KL loss weight (PM class-logit targets) for prototype teacher.")
    parser.add_argument('--fedcd_proto_teacher_noise_scale', type=float, default=1.0,
                        help="Noise scale for Gaussian prototype sampling.")
    parser.add_argument('--fedcd_proto_teacher_min_count', type=float, default=1.0,
                        help="Minimum class sample count required to use class prototype.")
    parser.add_argument('--fedcd_proto_teacher_client_samples', type=int, default=0,
                        help="Per-client max samples for prototype upload (0 = full local train set).")
    parser.add_argument('--fedcd_proto_teacher_confidence_weight', type=str2bool, default=True,
                        help="Enable confidence-weighted KL in prototype-teacher GM update.")
    parser.add_argument('--fedcd_proto_teacher_confidence_min', type=float, default=0.05,
                        help="Minimum per-sample KL weight for prototype teacher confidence weighting.")
    parser.add_argument('--fedcd_proto_teacher_confidence_power', type=float, default=1.0,
                        help="Exponent for prototype teacher confidence shaping.")
    parser.add_argument('--fedcd_search_enable', type=str2bool, default=False,
                        help="Enable FedCD run-level early stop for automated method/parameter search.")
    parser.add_argument('--fedcd_search_min_rounds', type=int, default=8,
                        help="Minimum evaluated rounds before FedCD early-stop checks are applied.")
    parser.add_argument('--fedcd_search_patience', type=int, default=6,
                        help="Stop when combined search score does not improve for this many eval rounds.")
    parser.add_argument('--fedcd_search_drop_patience', type=int, default=3,
                        help="Stop when both PM-local and GM-global drop simultaneously for this many eval rounds.")
    parser.add_argument('--fedcd_search_drop_delta', type=float, default=0.003,
                        help="Minimum per-round drop magnitude used by dual-drop early-stop rule.")
    parser.add_argument('--fedcd_search_score_gm_weight', type=float, default=0.75,
                        help="Weight of GM-only global accuracy in search score.")
    parser.add_argument('--fedcd_search_score_pm_weight', type=float, default=0.25,
                        help="Weight of PM-only local accuracy in search score.")
    parser.add_argument('--fedcd_search_score_eps', type=float, default=1e-4,
                        help="Minimum score improvement treated as progress in search mode.")
    parser.add_argument('--fedcd_search_min_pm_local_acc', type=float, default=0.55,
                        help="Minimum PM-local accuracy floor used by hard-fail early stop in search mode.")
    parser.add_argument('--fedcd_search_min_gm_global_acc', type=float, default=0.18,
                        help="Minimum GM-global accuracy floor used by hard-fail early stop in search mode.")
    parser.add_argument('--gm_model', type=str, default="VGG8",
                        help="FedCD GM model name")
    parser.add_argument('--pm_model', type=str, default="CNN",
                        help="FedCD PM model name")
    parser.add_argument('--fext_model', type=str, default="SmallFExt",
                        help="FedCD feature extractor model name")
    parser.add_argument('--fext_dim', type=int, default=512,
                        help="FedCD feature extractor output dimension")
    parser.add_argument('--eval_common_global', type=str2bool, default=True,
                        help="Evaluate personalized models on the same shared global test subset.")
    parser.add_argument('--global_test_samples', '--common_test_samples', dest='global_test_samples', type=int, default=0,
                        help="Number of samples for shared global test evaluation (0 = full union).")
    parser.add_argument('--common_eval_batch_size', type=int, default=256,
                        help="Batch size for shared global test evaluation.")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=str2bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=str2bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('-oom', "--avoid_oom", type=str2bool, default=True)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=str2bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=10000,
                        help="The threthold for droping slow clients")
    # pFedMe / PerAvg / FedProx / FedAMP / FedPHP / GPFL / FedCAC
    parser.add_argument('-bt', "--beta", type=float, default=0.0)
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-K', "--K", type=int, default=5,
                        help="Number of personalized training steps for pFedMe")
    parser.add_argument('-lrp', "--p_learning_rate", type=float, default=0.01,
                        help="personalized learning rate to caculate theta aproximately using K steps")
    # FedFomo
    parser.add_argument('-M', "--M", type=int, default=5,
                        help="Server only sends M client models to one client at each round")
    # FedMTL
    parser.add_argument('-itk', "--itk", type=int, default=4000,
                        help="The iterations for solving quadratic subproblems")
    # FedAMP
    parser.add_argument('-alk', "--alphaK", type=float, default=1.0, 
                        help="lambda/sqrt(GLOABL-ITRATION) according to the paper")
    parser.add_argument('-sg', "--sigma", type=float, default=1.0)
    # APFL / FedCross
    parser.add_argument('-al', "--alpha", type=float, default=1.0)
    # Ditto / FedRep
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    # MOON / FedCAC / FedLC
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    # FedBABU
    parser.add_argument('-fte', "--fine_tuning_epochs", type=int, default=10)
    # APPLE
    parser.add_argument('-dlr', "--dr_learning_rate", type=float, default=0.0)
    parser.add_argument('-L', "--L", type=float, default=1.0)
    # FedGen
    parser.add_argument('-nd', "--noise_dim", type=int, default=512)
    parser.add_argument('-glr', "--generator_learning_rate", type=float, default=0.005)
    parser.add_argument('-hd', "--hidden_dim", type=int, default=512)
    parser.add_argument('-se', "--server_epochs", type=int, default=1000)
    parser.add_argument('-lf', "--localize_feature_extractor", type=str2bool, default=False)
    # SCAFFOLD / FedGH
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    # FedALA
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    # FedKD
    parser.add_argument('-mlr', "--mentee_learning_rate", type=float, default=0.005)
    parser.add_argument('-Ts', "--T_start", type=float, default=0.95)
    parser.add_argument('-Te', "--T_end", type=float, default=0.98)
    # FedDBE
    parser.add_argument('-mo', "--momentum", type=float, default=0.1)
    parser.add_argument('-klw', "--kl_weight", type=float, default=0.0)

    # FedCross
    parser.add_argument('-fsb', "--first_stage_bound", type=int, default=0)
    parser.add_argument('-ca', "--fedcross_alpha", type=float, default=0.99)
    parser.add_argument('-cmss', "--collaberative_model_select_strategy", type=int, default=1)


    args = parser.parse_args()
    if args.algorithm == "FedCD":
        args.pm_model_name = args.pm_model
        args.model = args.gm_model

    # [New] Experiment Logging Setup
    partition_info = "unknown"
    alpha_info = ""
    # Prefer <repo>/fl_data, fallback to <repo_parent>/fl_data.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    fl_data_candidates = [
        os.path.join(repo_root, "fl_data"),
        os.path.abspath(os.path.join(repo_root, "..", "fl_data")),
    ]
    fl_data_root = next(
        (path for path in fl_data_candidates if os.path.isdir(path)),
        fl_data_candidates[0],
    )
    config_path = os.path.join(fl_data_root, args.dataset, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
                partition_info = cfg.get("partition", "unknown")
                if partition_info == "dir":
                    alpha_info = str(cfg.get("alpha", "unknown"))
        except Exception:
            pass

    date_str = time.strftime("%Y%m%d")
    timestamp = time.strftime("%H%M%S")
    gm_name = getattr(args, "gm_model", "none")
    pm_name = getattr(args, "pm_model_name", "none")
    fext_name = getattr(args, "fext_model", "none")
    
    # Structure: logs/FedCD/GM_{GM}_PM_{PM}_Fext_{Fext}/{partition}/{alpha if dir}/NC_{NC}/date_{date}/time_{time}
    path_parts = ["logs", args.algorithm, f"GM_{gm_name}_PM_{pm_name}_Fext_{fext_name}", partition_info]
    if partition_info == "dir" and alpha_info:
        path_parts.append(alpha_info)
    path_parts.extend([f"NC_{args.num_clients}", f"date_{date_str}", f"time_{timestamp}"])
    
    exp_dir = os.path.join(*path_parts)
    
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    # Update log paths FIRST
    args.log_usage_path = os.path.join(exp_dir, "acc.csv")
    print(f"\n[Info] Experiment logs will be saved to: {exp_dir}\n")

    # Save arguments AFTER updating paths
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # Validate dataset files under the resolved fl_data root.
    dataset_root = os.path.join(fl_data_root, args.dataset)
    train_file = os.path.join(dataset_root, "train", f"{args.num_clients - 1}.npz")
    test_file = os.path.join(dataset_root, "test", f"{args.num_clients - 1}.npz")
    if not (os.path.exists(train_file) and os.path.exists(test_file)):
        searched_roots = "\n".join(f"  - {path}" for path in fl_data_candidates)
        raise FileNotFoundError(
            "Dataset files not found in fl_data.\n"
            f"  dataset: {args.dataset}\n"
            f"  searched roots:\n{searched_roots}\n"
            f"  expected train: {train_file}\n"
            f"  expected test: {test_file}\n"
            "Please place the scenario under one of the fl_data roots above before running."
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    elif args.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = bool(getattr(args, "tf32", True))
        torch.backends.cudnn.allow_tf32 = bool(getattr(args, "tf32", True))
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    print("=" * 50)
    for arg in vars(args):
        print(arg, '=',getattr(args, arg))
    print("=" * 50)

    # with torch.profiler.profile(
    #     activities=[
    #         torch.profiler.ProfilerActivity.CPU,
    #         torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True, 
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
    #     ) as prof:
    # with torch.autograd.profiler.profile(profile_memory=True) as prof:
    run(args)

    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # print(f"\nTotal time cost: {round(time.time()-total_start, 2)}s.")
