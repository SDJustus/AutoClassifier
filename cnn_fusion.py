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
    parser.add_argument("--backbones", nargs="+", help="['vgg11', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121']")
    parser.add_argument("--seed", type=int, help="set seed for reproducability")
    parser.add_argument("--display", default=False, action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    #networks = ["vgg11", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101"]
    networks = cfg.backbones
    seed = cfg.seed
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_path = cfg.dataroot
    
    
    seed_torch(seed)
    _, _, inferenceLoader = get_train_valid_loader(1, dataset_path)
    
    
    aucroc_values = dict()
    model_predictions = dict()
    performances = dict()
    file_names = list()
    for network in networks:
        cfg.backbone = network
        cfg.name = network + "_" + str(seed)
        cfg.outf = network + "_" + str(seed)
        utils = Utils(1, device, cfg=cfg)
        print(f"Starting Backbone: {network} Seed: {str(seed)}")
        model = torch.load(os.path.join(network+"_"+str(seed), network+"test.pth"))
        model.to(device)
        before = cfg.display
        utils.cfg.display=False
        file_names, y_preds_after_threshold, y_trues = utils.inference(model, inferenceLoader, network=network, outf=cfg.outf)
        utils.cfg.display=before
        model_predictions[network] = np.array(y_preds_after_threshold)
        performances[network] = utils.get_performance(y_trues=y_trues, y_preds=y_preds_after_threshold)
        with open(os.path.join(network+"_"+str(seed), network+"_"+str(seed)+"_"+network+".txt"), "r") as file:
            regex_string = re.search(r"Inf.*auc\D,\s(\d\.\d+)", file.read())[1]
            aucroc_values[network] = float(regex_string)
            file.close()
    new_predictions = None
    for i in range(len(networks)):
        try:
            weight = aucroc_values[networks[i]]/sum(aucroc_values.values())
            print("weight", weight)
            if type(new_predictions) is np.array:
                print("i was here")
                np.concat(new_predictions, weight*model_predictions[networks[i]])
            else:
                new_predictions = np.array(weight*model_predictions[networks[i]])
            print("new_predictions", new_predictions)
            #print("file_names",file_names[networks[i]]==model_trues[networks[i+1]])
        except Exception as e:
            print(e)
    final_predictions = None
    print("final_preds", final_predictions)
    final_predictions = np.sum(new_predictions, axis=0)
    print("final_preds_after",final_predictions)
    y_preds = final_predictions
    performance, t, y_preds_after_threshold = utils.get_performance(y_trues=y_trues, y_preds=y_preds)
    print(performance)
    if cfg.display:
        utils.visualizer.plot_performance(epoch=1, performance=performance, tag="Fusion_Performance_AutoClassifier")
        utils.visualizer.plot_current_conf_matrix(epoch=1, cm=performance["conf_matrix"], tag="Fusion_Confusion_Matrix_AutoClassifier")
        utils.visualizer.plot_pr_curve(y_preds=y_preds, y_trues=y_trues, t=t, tag="Fusion_PR_Curve_AutoClassifier")
        utils.visualizer.plot_roc_curve(y_trues=y_trues, y_preds=y_preds, global_step=1, tag="ROC_Curve_Fusion")
    utils.write_inference_result(file_names=file_names, y_preds=y_preds_after_threshold, y_trues=y_trues, outf=os.path.join("classification_result_fusion_" + str(cfg.name) + "_" + network + ".json"))
        
    
    
    

    