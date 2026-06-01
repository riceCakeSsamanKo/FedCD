#!/usr/bin/env python
import copy
import torch
import torch.nn as nn
import argparse
import os
import time
import json
import warnings
import numpy as np
import random
import torchvision
import logging
import re

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
from flcore.servers.serverpac import FedPAC
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
from flcore.servers.servercwavg import cwFedAvg

from flcore.trainmodel.models import *

from flcore.trainmodel.bilstm import *
from flcore.trainmodel.resnet import *
from flcore.trainmodel.alexnet import *
from flcore.trainmodel.mobilenet_v2 import *
from flcore.trainmodel.transformer import *

from utils.result_utils import average_data
from utils.mem_utils import MemReporter
from utils.data_utils import read_client_data

logger = logging.getLogger()
logger.setLevel(logging.ERROR)

warnings.simplefilter("ignore")


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("true", "1", "yes", "y", "t"):
        return True
    if v in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def split_model(model):
    if hasattr(model, 'fc'):
        head = copy.deepcopy(model.fc)
        model.fc = nn.Identity()
    elif hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Sequential):
            last_linear_idx = None
            for idx in range(len(model.classifier) - 1, -1, -1):
                if isinstance(model.classifier[idx], nn.Linear):
                    last_linear_idx = idx
                    break
            if last_linear_idx is None:
                raise NotImplementedError("No linear classifier head found for splitting.")
            head = copy.deepcopy(model.classifier[last_linear_idx])
            model.classifier[last_linear_idx] = nn.Identity()
        else:
            head = copy.deepcopy(model.classifier)
            model.classifier = nn.Identity()
    else:
        raise NotImplementedError("Model structure not supported for splitting.")
    return BaseHeadSplit(model, head)


def build_client_class_distribution(dataset, num_clients, num_classes, few_shot=0):
    data_dist = np.zeros((num_clients, num_classes), dtype=np.float32)
    for client_id in range(num_clients):
        train_data = read_client_data(dataset, client_id, is_train=True, few_shot=few_shot)
        for _, label in train_data:
            label_idx = int(label.item()) if torch.is_tensor(label) else int(label)
            if 0 <= label_idx < num_classes:
                data_dist[client_id, label_idx] += 1
    return data_dist

def run(args):

    time_list = []
    reporter = MemReporter()
    model_str = args.model

    for i in range(args.prev, args.times):
        run_seed = args.seed + i
        set_global_seed(run_seed)
        print(f"\n============= Running time: {i}th =============")
        print(f"Using seed: {run_seed}")
        print("Creating server and clients ...")
        start = time.time()

        # Generate args.model
        if model_str == "MLR": # convex
            if "MNIST" in args.dataset:
                args.model = Mclr_Logistic(1*28*28, num_classes=args.num_classes) # .to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = Mclr_Logistic(3*32*32, num_classes=args.num_classes) # .to(args.device)
            else:
                args.model = Mclr_Logistic(60, num_classes=args.num_classes) # .to(args.device)

        elif model_str == "CNN": # non-convex
            if "MNIST" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024) # .to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600) # .to(args.device)
            elif "Omniglot" in args.dataset:
                args.model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=33856) # .to(args.device)
                # args.model = CifarNet(num_classes=args.num_classes).to(args.device)
            elif "Digit5" in args.dataset:
                args.model = Digit5CNN() # .to(args.device)
            else:
                args.model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=10816) # .to(args.device)

        elif model_str == "DNN": # non-convex
            if "MNIST" in args.dataset:
                args.model = DNN(1*28*28, 100, num_classes=args.num_classes) # .to(args.device)
            elif "Cifar10" in args.dataset:
                args.model = DNN(3*32*32, 100, num_classes=args.num_classes) # .to(args.device)
            else:
                args.model = DNN(60, 20, num_classes=args.num_classes) # .to(args.device)
        
        elif model_str == "ResNet18":
            args.model = torchvision.models.resnet18(pretrained=False, num_classes=args.num_classes) # .to(args.device)
            
            # args.model = torchvision.models.resnet18(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
            # args.model = resnet18(num_classes=args.num_classes, has_bn=True, bn_block_num=4).to(args.device)
        
        elif model_str == "ResNet10":
            args.model = resnet10(num_classes=args.num_classes) # .to(args.device)
        
        elif model_str == "ResNet34":
            args.model = torchvision.models.resnet34(pretrained=False, num_classes=args.num_classes) # .to(args.device)

        elif model_str == "AlexNet":
            args.model = alexnet(pretrained=False, num_classes=args.num_classes) # .to(args.device)
            
            # args.model = alexnet(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "GoogleNet":
            args.model = torchvision.models.googlenet(pretrained=False, aux_logits=False, 
                                                      num_classes=args.num_classes) # .to(args.device)
            
            # args.model = torchvision.models.googlenet(pretrained=True, aux_logits=False).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)

        elif model_str == "MobileNet":
            args.model = mobilenet_v2(pretrained=False, num_classes=args.num_classes) # .to(args.device)
            
            # args.model = mobilenet_v2(pretrained=True).to(args.device)
            # feature_dim = list(args.model.fc.parameters())[0].shape[1]
            # args.model.fc = nn.Linear(feature_dim, args.num_classes).to(args.device)
            
        elif model_str == "LSTM":
            args.model = LSTMNet(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes) # .to(args.device)

        elif model_str == "BiLSTM":
            args.model = BiLSTM_TextClassification(input_size=args.vocab_size, hidden_size=args.feature_dim, 
                                                   output_size=args.num_classes, num_layers=1, 
                                                   embedding_dropout=0, lstm_dropout=0, attention_dropout=0, 
                                                   embedding_length=args.feature_dim) # .to(args.device)

        elif model_str == "fastText":
            args.model = fastText(hidden_dim=args.feature_dim, vocab_size=args.vocab_size, num_classes=args.num_classes) # .to(args.device)

        elif model_str == "TextCNN":
            args.model = TextCNN(hidden_dim=args.feature_dim, max_len=args.max_len, vocab_size=args.vocab_size, 
                                 num_classes=args.num_classes) # .to(args.device)

        elif model_str == "Transformer":
            args.model = TransformerModel(ntoken=args.vocab_size, d_model=args.feature_dim, nhead=8, nlayers=2, 
                                          num_classes=args.num_classes, max_len=args.max_len) # .to(args.device)
        
        elif model_str == "VGG16":
            args.model = torchvision.models.vgg16(pretrained=True)
            in_features = args.model.classifier[6].in_features
            args.model.classifier[6] = nn.Linear(in_features, args.num_classes)
            # args.model = args.model.to(args.device)

        elif model_str == "VGG8":
            in_channels = 1 if "MNIST" in args.dataset or "Fashion" in args.dataset or "EMNIST" in args.dataset else 3
            args.model = VGG8(num_classes=args.num_classes, in_channels=in_channels)

        elif model_str == "AmazonMLP":
            args.model = AmazonMLP() # .to(args.device)

        elif model_str == "HARCNN":
            if args.dataset == 'HAR':
                args.model = HARCNN(9, dim_hidden=1664, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)) # .to(args.device)
            elif args.dataset == 'PAMAP2':
                args.model = HARCNN(9, dim_hidden=3712, num_classes=args.num_classes, conv_kernel_size=(1, 9), 
                                    pool_kernel_size=(1, 2)) # .to(args.device)

        else:
            raise NotImplementedError

        # args.model = args.model.to(args.device)
        print(args.model)

        # select algorithm
        if args.algorithm == "FedAvg":
            args.model = split_model(args.model)
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
            args.model = split_model(args.model)
            server = FedPer(args, i)

        elif args.algorithm == "Ditto":
            server = Ditto(args, i)

        elif args.algorithm == "FedRep":
            args.model = split_model(args.model)
            server = FedRep(args, i)

        elif args.algorithm == "FedPHP":
            args.model = split_model(args.model)
            server = FedPHP(args, i)

        elif args.algorithm == "FedBN":
            server = FedBN(args, i)

        elif args.algorithm == "FedROD":
            args.model = split_model(args.model)
            server = FedROD(args, i)

        elif args.algorithm == "FedProto":
            args.model = split_model(args.model)
            server = FedProto(args, i)

        elif args.algorithm == "FedDyn":
            server = FedDyn(args, i)

        elif args.algorithm == "MOON":
            args.model = split_model(args.model)
            server = MOON(args, i)

        elif args.algorithm == "FedBABU":
            args.model = split_model(args.model)
            server = FedBABU(args, i)

        elif args.algorithm == "APPLE":
            server = APPLE(args, i)

        elif args.algorithm == "FedGen":
            args.model = split_model(args.model)
            server = FedGen(args, i)

        elif args.algorithm == "SCAFFOLD":
            server = SCAFFOLD(args, i)

        elif args.algorithm == "FD":
            server = FD(args, i)

        elif args.algorithm == "FedALA":
            server = FedALA(args, i)

        elif args.algorithm == "FedPAC":
            args.model = split_model(args.model)
            server = FedPAC(args, i)

        elif args.algorithm == "LG-FedAvg":
            args.model = split_model(args.model)
            server = LG_FedAvg(args, i)

        elif args.algorithm == "FedGC":
            args.model = split_model(args.model)
            server = FedGC(args, i)

        elif args.algorithm == "FML":
            server = FML(args, i)

        elif args.algorithm == "FedKD":
            args.model = split_model(args.model)
            server = FedKD(args, i)

        elif args.algorithm == "FedPCL":
            args.model.fc = nn.Identity()
            server = FedPCL(args, i)

        elif args.algorithm == "FedCP":
            args.model = split_model(args.model)
            server = FedCP(args, i)

        elif args.algorithm == "GPFL":
            args.model = split_model(args.model)
            server = GPFL(args, i)

        elif args.algorithm == "FedNTD":
            server = FedNTD(args, i)

        elif args.algorithm == "FedGH":
            args.model = split_model(args.model)
            server = FedGH(args, i)

        elif args.algorithm == "FedDBE":
            args.model = split_model(args.model)
            server = FedDBE(args, i)

        elif args.algorithm == 'FedCAC':
            server = FedCAC(args, i)

        elif args.algorithm == 'PFL-DA':
            args.model = split_model(args.model)
            server = PFL_DA(args, i)

        elif args.algorithm == 'FedLC':
            args.model = split_model(args.model)
            server = FedLC(args, i)

        elif args.algorithm == 'FedAS':

            args.model = split_model(args.model)
            server = FedAS(args, i)
            
        elif args.algorithm == "FedCross":
            server = FedCross(args, i)

        elif args.algorithm == "cwFedAvg":
            args.add_cw = True
            args.model = split_model(args.model)
            if args.partial_layer_train:
                layer_names = [name for name, _ in args.model.named_children()]
                args.layer_groups = {
                    "common": layer_names[:-args.cw_layer_num],
                    "cw": layer_names[-args.cw_layer_num:],
                }
            if not hasattr(args, "data_dist"):
                args.data_dist = build_client_class_distribution(
                    args.dataset, args.num_clients, args.num_classes, args.few_shot
                ).tolist()
            server = cwFedAvg(args, i)

        else:
            raise NotImplementedError

        server.train()

        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    

    # Global average
    average_data(dataset=args.dataset, algorithm=args.algorithm, goal=args.goal, times=args.times)

    print("All done!")

    reporter.report()


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="VGG8")
    parser.add_argument('-lbs', "--batch_size", type=int, default=128)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=100)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=1, 
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('--seed', type=int, default=0,
                        help="Base random seed. Run i uses seed+i.")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('--eval_common_global', type=str2bool, default=True,
                        help="Evaluate each client model on one shared common global test subset.")
    parser.add_argument('--global_test_samples', '--common_test_samples', dest='global_test_samples', type=int, default=0,
                        help="Number of samples for shared global test evaluation (0 = full union).")
    parser.add_argument('--common_eval_batch_size', type=int, default=256,
                        help="Batch size for common global test evaluation.")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-vs', "--vocab_size", type=int, default=80, 
                        help="Set this for text tasks. 80 for Shakespeare. 32000 for AG_News and SogouNews.")
    parser.add_argument('-ml', "--max_len", type=int, default=200)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    # practical
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=bool, default=False,
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
    parser.add_argument('-lf', "--localize_feature_extractor", type=bool, default=False)
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

    # cwFedAvg
    parser.add_argument('-cw', "--add_cw", action='store_true')
    parser.add_argument('-wdr', "--add_wdr", action='store_true')
    parser.add_argument('-dlo', "--decision_layer_only", action='store_true')
    parser.add_argument('-plt', "--partial_layer_train", action='store_true')
    parser.add_argument('-spl', "--split_train", action='store_true')
    parser.add_argument('-hlr', "--head_lr", type=float, default=0.005)
    parser.add_argument('-hbs', "--head_bs", type=int, default=10)
    parser.add_argument('-ncw', "--cw_layer_num", type=int, default=1)
    parser.add_argument('-apr', "--add_proto", action='store_true')
    parser.add_argument('-gt', "--use_true_dist", action='store_true')
    parser.add_argument('-be', "--batch_eval", action='store_true')
    parser.add_argument('-clw', "--clip_weight", action='store_true')
    parser.add_argument('-beid', "--batch_eval_id", type=int, default=0)
    parser.add_argument('-wd', "--weight_decay", type=float, default=10.0)


    args = parser.parse_args()

    # [New] Experiment Logging Setup (FedCD Style)
    partition_info = "unknown"
    alpha_info = ""
    # Parse info from dataset name (e.g., Cifar10_dir0.1_nc20)
    if "pat" in args.dataset:
        partition_info = "pat"
    elif "dir" in args.dataset:
        partition_info = "dir"
        # Extract alpha (e.g., dir0.1 -> 0.1)
        match = re.search(r"dir([0-9.]+)", args.dataset)
        if match:
            alpha_info = match.group(1)

    date_str = time.strftime("%Y%m%d")
    timestamp = time.strftime("%H%M%S")
    model_name = args.model if isinstance(args.model, str) else "Model"
    
    # Always write logs to the repository-root logs directory,
    # even when launched from system/ (e.g., `cd system && python main.py`).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    fl_data_candidates = [
        os.environ.get("FL_DATA_ROOT", ""),
        os.path.abspath(os.path.join(repo_root, "..", "fl_data")),
        os.path.abspath(os.path.join(repo_root, "..", "..", "fl_data")),
        os.path.join(repo_root, "fl_data"),
    ]
    for fl_data_root in fl_data_candidates:
        if not fl_data_root:
            continue
        config_path = os.path.join(fl_data_root, args.dataset, "config.json")
        if not os.path.exists(config_path):
            continue
        try:
            with open(config_path, "r") as f:
                cfg = json.load(f)
            cfg_partition = cfg.get("partition", partition_info)
            if cfg.get("splitgp_rho", None) is not None:
                partition_info = f"splitgp_rho{float(cfg.get('splitgp_rho')):.1f}"
                alpha_info = ""
                args.eval_common_global = False
            elif cfg_partition == "dir":
                partition_info = "dir"
                alpha_info = str(cfg.get("alpha", alpha_info or "unknown"))
            elif cfg_partition:
                partition_info = cfg_partition
            break
        except Exception:
            pass

    dataset_name = args.dataset.split("_")[0] if isinstance(args.dataset, str) and len(args.dataset) > 0 else "UnknownDataset"
    dataset_name = dataset_name.replace("/", "-")
    if dataset_name == "Cifar10":
        dataset_name = "cifar10"

    # Structure:
    # {repo_root}/logs/{dataset}/{Algorithm}/GM_{Model}/{partition}/{alpha if dir}/NC_{NC}/date_{date}/time_{time}
    path_parts = [repo_root, "logs", dataset_name, args.algorithm, f"GM_{model_name}", partition_info]
    if partition_info == "dir" and alpha_info:
        path_parts.append(alpha_info)
    path_parts.extend([f"NC_{args.num_clients}", f"date_{date_str}"])

    base_exp_dir = os.path.join(*path_parts, f"time_{timestamp}")
    goal_token = re.sub(r"[^A-Za-z0-9_.=-]+", "-", str(args.goal)).strip("-")[:80]
    seed_token = f"seed{getattr(args, 'seed', 'unknown')}"
    suffix_candidates = [
        "",
        f"_{seed_token}",
        f"_{goal_token}" if goal_token else f"_{seed_token}_{os.getpid()}",
        f"_{seed_token}_pid{os.getpid()}",
    ]

    args.exp_dir = None
    for suffix in suffix_candidates:
        exp_dir = base_exp_dir + suffix
        try:
            os.makedirs(exp_dir, exist_ok=False)
            args.exp_dir = exp_dir
            break
        except FileExistsError:
            continue

    if args.exp_dir is None:
        # Last-resort uniqueness for heavily parallel launches in the same second.
        for idx in range(2, 10000):
            exp_dir = f"{base_exp_dir}_{seed_token}_pid{os.getpid()}_{idx}"
            try:
                os.makedirs(exp_dir, exist_ok=False)
                args.exp_dir = exp_dir
                break
            except FileExistsError:
                continue
        if args.exp_dir is None:
            raise RuntimeError(f"Could not create a unique log directory under {base_exp_dir}")
    
    args.log_path = os.path.join(args.exp_dir, "acc.csv")
    print(f"\n[Info] Experiment logs will be saved to: {args.exp_dir}\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"

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
