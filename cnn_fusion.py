import argparse
import random
import numpy as np
import torch
import os
import re
from split_train import get_train_valid_loader
from utils import Utils
    
def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def parse_args():
    parser = argparse.ArgumentParser(prog="AutoClassifier")
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--name", required=True, type=str, help="Name for the train run")
    parser.add_argument("--size",
                        default="256",
                        type=int,
                        help="image size [default=256x256]")
    parser.add_argument("--backbones", nargs="+", help="['vgg11', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121']")
    parser.add_argument("--seed", type=int, help="set seed for reproducability")
    parser.add_argument("--batchsize", type=int, default=32, help="batchsize....")
    parser.add_argument("--outf", type=str, default="./output", help="dir to write results in!")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    #networks = ["vgg11", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101"]
    networks = cfg.backbones
    seed = cfg.seed
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_path = cfg.dataroot
    utils = Utils(1, device, cfg=cfg)
    outf = cfg.outf
    
    
    _, _, inferenceLoader = get_train_valid_loader(1, dataset_path)
    seed_torch(seed)
    models = list()
    aucroc_values = dict()
    file_names = dict()
    model_predictions = dict()
    model_trues = dict()
    performances = dict()
    for network in networks:
        print(f"Starting Backbone: {network} Seed: {str(seed)}")
        model = torch.load(os.path.join(network+"_"+str(seed)), network+str(seed)+"test.pth")
        model.to(device)
        f_names, y_preds_after_threshold, y_trues = utils.inference(model, inferenceLoader, network=network, outf=outf)
        file_names[network] = f_names
        model_predictions[network] = y_preds_after_threshold
        model_trues[network] = y_trues
        performances[network] = utils.get_performance(y_trues=y_trues, y_preds=y_preds_after_threshold)
        with open(os.path.join(network+"_"+str(seed), network+"_"+str(seed)+"_"+network+str(seed)+".txt"), "r") as file:
            aucroc_values[network] = re.search(r"Inf*.*auc', +(\d.\d+)'", file.read())
            file.close()
    for i in range(len(networks)):
        try:
            
            voting =  aucroc_values[networks[i]]/sum(aucroc_values.values())
            
            print("model_trues",model_trues[networks[i]]==model_trues[networks[i+1]])
            print("file_names",file_names[networks[i]]==model_trues[networks[i+1]])
        except Exception as e:
            print(e)
    voting = dict()
    for backbone, aucroc in aucroc_values.items():
        voting[backbone] = aucroc/sum(aucroc_values.values())
    
    

    