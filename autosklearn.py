import argparse
from itertools import chain
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from split_train import get_train_valid_loader
import h2o
from h2o.automl import H2OAutoML
import utils
from split_train import get_train_valid_loader
import time

def parse_args():
    parser = argparse.ArgumentParser(prog="AutoClassifier")
    parser.add_argument("--dataroot", required=True, type=str)
    parser.add_argument("--backbone", type=str, help="['vgg11', 'vgg16', 'vgg19', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'densenet121']")
    parser.add_argument("--seed", type=int, help="set seed for reproducability")
    parser.add_argument("--display", default=False, action="store_true")
    parser.add_argument("--batchsize", type=int, default=32, help="batchsize....")
    parser.add_argument("--inference_only", action="store_true", default=False, help="if only inference should be performed")
    parser.add_argument("--outf", type=str, default="./output", help="dir to write results in!")
    parser.add_argument("--decision_threshold", type=float, default=None, help="set the decision threshold for the anomaly score manually. If not set, it is computed by AUROC-Metric")
    
    return parser.parse_args()



# Generate dataset features, removing VGG16 classification component
def generateDatasetFeatures(network, cfg, device):
    
    trainLoader, testLoader, inferenceLoader = get_train_valid_loader(cfg.batchsize, cfg.dataroot)
    model = torch.load(os.path.join(network+"_"+str(cfg.seed), network+"test.pth"))
    print(model)
    num_ftrs = None
    if "resnet" in network:
        num_ftrs = model.fc[0].in_features
    elif "vgg" in network:
        num_ftrs = model.classifier[6][0].in_features
    else:
        raise NotImplementedError(network + "not implemented yet.")
    model.to(device)
    model.fc = nn.Identity()

    print("Prepare Inference csv-file")
    f = open('./dataset/'+network+'_inference.csv','w')
    #f.write('x\ty\n')

    inferenceTime = []
    with torch.no_grad():
        model.eval()
        fnames_inf = [] 
        for data in tqdm(inferenceLoader):
            inputs, labels, fnames = data
            fnames_inf.append(fnames)
            inputs = inputs.to(device)

            #stBatch = time.time()
            outputs = model(inputs)
            #etBatch = time.time()-stBatch
            #inferenceTime.append(etBatch)
            for output, label in zip(outputs.cpu().numpy(), labels.numpy()):
                #print(np.array2string(output)+"\t"+np.array2string(label))
                for value in output:
                    f.write(str(value)+",")
                f.write(str(label))
                #f.write(str(output.tolist())+","+str(label)) #Give your csv text here.
                f.write("\n")
    f.close()
    #inferenceTimeMean = [i/batch_size for i in inferenceTime] # list batches time
    #print (inferenceTimeMean)
    #print(f'Inference Time Mean: {np.mean(inferenceTimeMean)}, STD:{np.std(inferenceTimeMean)}')
    
    print("Prepare Test csv-file")
    f = open('./dataset/'+network+'_test.csv','w')
    with torch.no_grad():
        for data in tqdm(testLoader):
            inputs, labels, _ = data
            inputs = inputs.to(device)

            outputs = model(inputs)
            for output, label in zip(outputs.cpu().numpy(), labels.numpy()):
                for value in output:
                    f.write(str(value)+",")
                f.write(str(label))
                f.write("\n")
    f.close()

    print("Prepare Train csv-file")
    f = open('./dataset/'+network+'_train.csv','w')
    with torch.no_grad():
        for data in tqdm(trainLoader):
            inputs, labels, _ = data
            inputs = inputs.to(device)

            outputs = model(inputs)
            for output, label in zip(outputs.cpu().numpy(), labels.numpy()):
                for value in output:
                    f.write(str(value)+",")
                f.write(str(label))
                f.write("\n")
    f.close()
    return fnames_inf, num_ftrs



if __name__ == "__main__":
    h2o.init()
    cfg = parse_args()
    train_time = None
    inf_time = None
    train_start = time.time()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    network= cfg.backbone
    cfg.name = network + "_" + str(cfg.seed)
    cfg.outf = "AUTOML_" + str(cfg.seed)
    if not os.path.isdir(cfg.outf): os.mkdir(cfg.outf)
    utils = utils.Utils(cfg.batchsize, device, cfg=cfg)
    file_names, num_ftrs = generateDatasetFeatures(network, cfg=cfg, device=device)
    print(file_names)
    print("[Starting Problem")
    # put path for the newly datasets generated before
    
    x_train = h2o.import_file('./dataset/'+network+'_train.csv')
    x_test = h2o.import_file('./dataset/'+network+'_test.csv')
    x = x_train.columns
    y = "C" + str(num_ftrs+1)
    x.remove(y)
    
    x_inference = h2o.import_file('./dataset/'+network+'_inference.csv')
    y_inference = x_inference['C2049'] #predictions
    
    #x_train[y] = x_train[y].asfactor()
    #x_val[y] = x_val[y].asfactor()
    #x_test[y] = x_test[y].asfactor()

    if cfg.inference_only:
        path = os.path.join(cfg.outf, "AutoML"+str(cfg.seed)+".pth")
        modelfile = [os.path.join(path,f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))][0]
        aml = h2o.load_model(modelfile)
    else:
        aml = H2OAutoML(max_models = 30, max_runtime_secs=int(3600*2), seed = cfg.seed) #each problem will be searched for 2 hours
        aml.train(y = y, training_frame = x_train, validation_frame=x_test)
        train_time = time.time()-train_start
        lb = aml.leaderboard
        print(lb.head())
        lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
        print(lb)

    if not cfg.inference_only:
        h2o.save_model(aml.leader, path="AutoML"+str(cfg.seed) + ".pth")
    for x, mode in zip([x_test, x_inference], ["test", "inference"]):
        inf_start = time.time()
        preds = aml.predict(x)
        print("Predictions")
        inf_time = time.time()-inf_start
        y_trues = np.rint(np.array(h2o.as_list(x[y]))).astype(int)
        y_preds = np.array(h2o.as_list(preds))
        print("Metrics [...]")
        
        y_preds = list(chain(*y_preds))
        y_trues = list(chain(*y_trues))
        file_names = list(chain(*file_names))
        print (f'Train time: {train_time} secs')
        print (f'Inference time: {inf_time} secs')
        print (f'Inference time / individual: {inf_time/len(y_trues)} secs')
        
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
                utils.write_inference_result(file_names=file_names, y_preds=y_preds_auc, y_trues=y_trues, outf=os.path.join(cfg.outf,"classification_result_" + str(cfg.name) + "_" + mode + ".json"))
                if cfg.decision_threshold:
                    utils.write_inference_result(file_names=file_names, y_preds=y_preds_man, y_trues=y_trues, outf=os.path.join(cfg.outf,"classification_result_" + str(cfg.name) + "_" + mode + "_man.json"))
                
        with open(os.path.join(cfg.outf, "AutoML_" + str(cfg.seed) + "_" + mode + ".txt"), "a") as f:
            f.write(f'Inf Performance: {str(performance)}, Inf_times: {str(inf_time)}')
            f.close()

