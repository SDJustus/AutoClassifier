import argparse
import random
import time
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
    parser.add_argument("--outf", type=str, default="./output", help="dir to write results in!")
    parser.add_argument("--inference_only", action="store_true", default=False, help="do only inference")
    parser.add_argument("--save_anomaly_map", default=False, action="store_true", help="if the anomaly maps should be saved")
    parser.add_argument("--decision_threshold", type=float, default=None, help="set the decision threshold for the anomaly score manually. If not set, it is computed by AUROC-Metric")
    return parser.parse_args()


if __name__ == "__main__":
    cfg = parse_args()
    #networks = ["vgg11", "vgg16", "vgg19", "resnet18", "resnet34", "resnet50", "resnet101"]
    networks = cfg.backbones
    seed = cfg.seed
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dataset_path = cfg.dataroot
    inf_time = None
    
    
    seed_torch(seed)
    _, test_loader, inference_loader = get_train_valid_loader(1, dataset_path)
    
    
    
    
    for loader, mode in zip([test_loader, inference_loader], ["test", "inference"]):
        new_predictions = None
        weight_dict = dict()
        inf_start = time.time()
        aucroc_values = dict()
        model_predictions = dict()
        performances = dict()
        file_names = list()
        for i, network in enumerate(networks):
            with open(os.path.join(network+"_"+str(seed), network+"_"+str(seed)+"_"+network+".txt"), "r") as file:
                regex_string = re.search(r"Inf.*auc\D,\s(\d\.\d+)", file.read())[1]
                aucroc_values[network] = float(regex_string)
                file.close()
        for i, network in enumerate(networks):
            weight = aucroc_values[network]/sum(aucroc_values.values())
            weight_dict[network] = weight
            print(network,"weight", weight)
            cfg.backbone = network
            cfg.name = network + "_" + str(seed)
            cfg.outf = "cnn_fusion_" + str(seed)
            if not os.path.isdir(cfg.outf): os.mkdir(cfg.outf)
            utils = Utils(1, device, cfg=cfg)
            print(f"Starting Backbone: {network} Seed: {str(seed)}")
            model = torch.load(os.path.join(network+"_"+str(seed), network+"test.pth"))
            model.to(device)
            before = cfg.display
            utils.cfg.display=False
            file_names, y_preds_after_threshold, y_trues = utils.inference(model, loader, network=network, outf=cfg.outf)
            model_predictions[network] = np.array(y_preds_after_threshold)
            performances[network] = utils.get_performance(y_trues=y_trues, y_preds=y_preds_after_threshold)
            utils.cfg.display=before
            
            if type(new_predictions) is np.ndarray:
                #print("i was here")
                new_predictions = np.vstack((new_predictions, weight*model_predictions[network]))
            else:
                new_predictions = np.array(weight*model_predictions[network])
        
        with open(os.path.join(cfg.outf, "model_weights.txt"), "a") as f:
            f.write(weight_dict)
            f.write("\n")
            f.close() 
        #print("final_preds", final_predictions)
        final_predictions = np.sum(new_predictions, axis=0)
        #print("final_preds_after",final_predictions)
        y_preds = final_predictions
        inf_time = time.time()-inf_start
        print (f'Inference time_fusion: {inf_time} secs')
        print (f'Inference time / individual_fusion: {inf_time/len(y_trues)} secs')
        performance, t, y_preds_man, y_preds_auc = utils.get_performance(y_preds=y_preds, y_trues=y_trues, manual_threshold=cfg.decision_threshold)
        print(performance)
        
        if cfg.display:
            utils.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["threshold"], global_step=1, save_path=os.path.join(cfg.outf,"histogram_" + mode + ".png"), tag="Histogram_" + mode)
            utils.visualizer.plot_performance(epoch=1, performance=performance, tag="" + mode + "_Performance_AutoClassifier")
            utils.visualizer.plot_current_conf_matrix(1, performance["conf_matrix"], save_path=os.path.join(cfg.outf, "conf_matrix_" + mode + ".png"))
            if cfg.decision_threshold:
                utils.visualizer.plot_current_conf_matrix(2, performance["conf_matrix_man"], save_path=os.path.join(cfg.outf, "conf_matrix_man" + mode + ".png"))
                utils.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["manual_threshold"], global_step=2, save_path=os.path.join(cfg.outf, "histogram_man" + mode + ".png"), tag="Histogram_" + mode + "_man")
            utils.visualizer.plot_roc_curve(y_trues=y_trues, y_preds=y_preds, global_step=1, tag="ROC_Curve", save_path=os.path.join(cfg.outf, "roc_" + mode + ".png"))
            
            if mode == "inference":
                utils.write_inference_result(file_names=file_names, y_preds=y_preds_auc, y_trues=y_trues, outf=os.path.join(cfg.outf,"classification_result_" + str(cfg.name) + "_" + network + ".json"))
                if cfg.decision_threshold:
                    utils.write_inference_result(file_names=file_names, y_preds=y_preds_man, y_trues=y_trues, outf=os.path.join(cfg.outf,"classification_result_" + str(cfg.name) + "_" + network + "_man.json"))
        cfg.decision_threshold = performance["threshold"] if cfg.decision_threshold is not None else cfg.decision_threshold
        with open(os.path.join(cfg.outf, "fusion" + str(cfg.seed) + "_" + mode + ".txt"), "a") as f:
            f.write(f'Inf Performance: {str(performance)}, Inf_times: {str(sum(inf_time))}')
            f.close()
        
    
    

    