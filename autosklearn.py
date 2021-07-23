import argparse
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
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    network= cfg.backbone
    cfg.name = network + "_" + str(cfg.seed)
    cfg.outf = network + "_" + str(cfg.seed)
    utils = utils.Utils(cfg.batchsize, device, cfg=cfg)
    file_names, num_ftrs = generateDatasetFeatures(network, cfg=cfg, device=device)
    print(file_names)
    print("[Starting Problem")
    # put path for the newly datasets generated before
    if not cfg.inference_only:
        x_train = h2o.import_file('./dataset/'+network+'_train.csv')
        x_test = h2o.import_file('./dataset/'+network+'_test.csv')
        x = x_train.columns
        y = "C" + str(num_ftrs+1)
        x.remove(y)
        print(x)
    x_inference = h2o.import_file('./dataset/'+network+'_inference.csv')
    y_inference = x_inference['C2049'] #predictions
    
    #x_train[y] = x_train[y].asfactor()
    #x_val[y] = x_val[y].asfactor()
    #x_test[y] = x_test[y].asfactor()

    if cfg.inference_only:
        aml = h2o.load_model("AutoML"+str(cfg.seed)+".pth")
    else:
        aml = H2OAutoML(max_models = 30, max_runtime_secs=int(3600*2), seed = cfg.seed) #each problem will be searched for 2 hours
        aml.train(y = y, training_frame = x_train, validation_frame=x_test)

        lb = aml.leaderboard
        print(lb.head())
        lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
        print(lb)

    startTime = time.time()
    preds = aml.predict(x_inference)
    print("Predictions")
    endTime = time.time()-startTime
    print (f'Prediction time: {endTime} secs')
    print (f'Prediction time / individual: {endTime/173} secs')
    print(preds)
    
    if not cfg.inference_only:
        h2o.save_model(aml.leader, path="AutoML"+str(cfg.seed) + ".pth")

    y_trues = np.rint(np.array(h2o.as_list(x_inference[y]))).astype(int)
    y_preds = np.array(h2o.as_list(preds))
    print("Metrics [...]")
    
    y_preds = [y[0] for y in y_preds]
    y_trues = [y[0] for y in y_trues]
    performance, t, y_preds_after_threshold = utils.get_performance(y_trues=y_trues, y_preds=y_preds)
    print(performance)
    
    if cfg.display:
        utils.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["threshold"], global_step=1, save_path=os.path.join("histogram_automl_" + str(cfg.name) + "_" + network + ".csv"), tag="Histogram_AutoML_"+str(cfg.name))
        utils.visualizer.plot_performance(epoch=1, performance=performance, tag="AutoML_Performance_AutoClassifier")
        utils.visualizer.plot_current_conf_matrix(epoch=1, cm=performance["conf_matrix"], tag="AutoML_Confusion_Matrix_AutoClassifier")
        utils.visualizer.plot_pr_curve(y_preds=y_preds, y_trues=y_trues, t=t, tag="AutoML_PR_Curve_AutoClassifier")
        utils.visualizer.plot_roc_curve(y_trues=y_trues, y_preds=y_preds, global_step=1, tag="ROC_Curve_AutoML")
    utils.write_inference_result(file_names=file_names, y_preds=y_preds_after_threshold, y_trues=y_trues, outf=os.path.join("classification_result_automl_" + str(cfg.name) + "_" + network + ".json"))
        
    

