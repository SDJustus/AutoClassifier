from torchvision.datasets import vision
import argparse
import torch
import numpy as np
from split_train import get_train_valid_loader
from sklearn import metrics

from utils import Utils
import time
import copy
import random


def parse_args():
    parser = argparse.ArgumentParser(prog="AutoClassifier")
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--name", required=True, type=str, help="Name for the train run")
    parser.add_argument("--display", default=False, action="store_true")
    parser.add_argument("--size",
                        default="256",
                        type=int,
                        help="image size [default=256x256]")
    parser.add_argument("--epochs", type=int, default=50, help="epochs to train")
    parser.add_argument("--backbone", type=str, help="['vgg11', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121']")
    parser.add_argument("--seed", type=int, help="set seed for reproducability")
    parser.add_argument("--batchsize", type=int, default=32, help="batchsize....")
    return parser.parse_args()

def seed_torch(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":  
    # Iterate over all problems
    cfg = parse_args()
    path = cfg.dataroot
    input_shape = cfg.size
    batch_size = cfg.batchsize
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    epochs = cfg.epochs
    utils = Utils(batch_size, device, cfg=cfg)
    network = cfg.backbone
    num_output_neurons = 1
    
    
   
    seed_torch(cfg.seed)
    
    # Create model
    print("[Creating the model ...]")
    print(f"{network}")
    
    # Model selection
    model, _ = utils.initializeModel(network, num_output_neurons, True)
    model.to(device)

    trainLoader, testLoader, inferenceLoader = get_train_valid_loader(batch_size, path)

    ## Weight dataset loss
    #weights = torch.tensor([1., 10.])
    criterion = utils.getCrossEntropyLoss(binary=(True if num_output_neurons == 1 else False))
    optimizer_conv = utils.getSGDOptimizer(model)

    regular_train_auroc, regular_train_loss = [], []
    regular_val_auroc, regular_val_loss = [], []
    # Store model in best val acc, best val loss
    best_model_wts = copy.deepcopy(model.state_dict())
    best_model_lowest_loss = copy.deepcopy(model.state_dict())
    best_auroc_val = [0.0, 0] # accuracy, epoch
    best_loss_val = [20.0, 0] # loss, epoch

    # Time for training
    startTimeTrain = time.time()

    print("### Start Training [...] ###")
    for epoch in range(0,epochs):
        # Train
        model, train_performance, train_loss, predictions = utils.train(model, trainLoader, criterion, optimizer_conv, epoch)
        train_auroc = train_performance["auc"]
        regular_train_auroc.append(train_auroc)
        regular_train_loss.append(train_loss)

        # Validation
        val_performance, val_loss, predictions = utils.test(model, testLoader, criterion, epoch)
        val_auroc = val_performance["auc"]
        regular_val_auroc.append(val_auroc)
        regular_val_loss.append(val_loss)

        if val_auroc > best_auroc_val[0]: # store best model so far, for later, based on best val acc
            best_model_wts = copy.deepcopy(model.state_dict())
            best_auroc_val[0], best_auroc_val[1] = val_auroc, epoch
        if val_loss < best_loss_val[0]: # store best model according to loss
            best_model_lowest_loss = copy.deepcopy(model.state_dict())
            best_loss_val[0], best_loss_val[1] = val_loss, epoch

    endTimeTrain = time.time()

    model.load_state_dict(best_model_wts)
    utils.inference(model, inferenceLoader, network=network)
    torch.save(model, "./" + str(network) + "test.pth")