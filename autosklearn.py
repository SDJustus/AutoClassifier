from typing import OrderedDict
import torch
import torch.nn as nn
import numpy as np
from split_train import get_train_valid_loader
from sklearn import metrics
# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pandas
import h2o
from h2o.automl import H2OAutoML
import pandas as pd

import utils
from split_train import get_train_valid_loader
import time

path         = "../../isi-diploma-model-tests/skip-ganomaly/data/custom_dataset_adjusted_deep/"
path_models  = "./resnet50test.pth"
batch_size   = 10
device       = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
utils        = utils.Utils(batch_size, device)

network = 'resnet50'
model = utils.initializeModel(network, 2)


# Generate dataset features, removing VGG16 classification component
def generateDatasetFeatures(network):
    
    trainLoader, valLoader, testLoader = get_train_valid_loader(batch_size, path)
    model = torch.load(path_models)
    model.to(device)
    model.fc = nn.Identity()

    print("Test")
    f = open('./dataset/'+network+'_test.csv','w')
    #f.write('x\ty\n')

    inferenceTime = []
    with torch.no_grad():
        model.eval()
        for data in testLoader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            #stBatch = time.time()
            outputs = model(inputs)
            #etBatch = time.time()-stBatch
            #inferenceTime.append(etBatch)
            for output, label in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
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
    
    print("Validation")
    f = open('./dataset/'+network+'_validation.csv','w')
    with torch.no_grad():
        for data in valLoader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            for output, label in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                for value in output:
                    f.write(str(value)+",")
                f.write(str(label))
                f.write("\n")
    f.close()

    print("Train")
    f = open('./dataset/'+network+'_train.csv','w')
    with torch.no_grad():
        for data in trainLoader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            for output, label in zip(outputs.cpu().numpy(), labels.cpu().numpy()):
                for value in output:
                    f.write(str(value)+",")
                f.write(str(label))
                f.write("\n")
    f.close()



if __name__ == "__main__":
    h2o.init()
    #generateDatasetFeatures(network)
    print("[Starting Problem")
    # put path for the newly datasets generated before
    network= 'resnet50'
    x_train = h2o.import_file('./dataset/'+network+'_train.csv')
    x_val = h2o.import_file('./dataset/'+network+'_validation.csv')
    x_test = h2o.import_file('./dataset/'+network+'_test.csv')
    y_test = x_test['C2049'] #predictions
    x = x_train.columns
    print(x_test)
    y = 'C2049'
    x.remove(y)
    #x_train[y] = x_train[y].asfactor()
    #x_val[y] = x_val[y].asfactor()
    #x_test[y] = x_test[y].asfactor()


    aml = H2OAutoML(max_models = 30, max_runtime_secs=int(3600*2), seed = 1) #each problem will be searched for 2 hours
    aml.train(y = y, training_frame = x_train, validation_frame=x_val)

    lb = aml.leaderboard
    print(lb.head())

    startTime = time.time()
    preds = aml.predict(x_test)
    print("Predictions")
    endTime = time.time()-startTime
    print (f'Prediction time: {endTime} secs')
    print (f'Prediction time / individual: {endTime/173} secs')
    print(preds)
    print()
    lb = h2o.automl.get_leaderboard(aml, extra_columns = 'ALL')
    print(lb)
    #h2o.save_model(aml.leader, path = "./AutoML_models/problem"+str(problem)+"/")

    true_label = np.rint(np.array(h2o.as_list(x_test[y]))).astype(int)
    predictions = np.array(h2o.as_list(preds))
    print("Metrics [...]")

    fpr, tpr, t = metrics.roc_curve(true_label, predictions, pos_label=0)
    roc_score = metrics.auc(fpr, tpr)
    
    #Threshold
    i = np.arange(len(tpr))
    roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(t, index=i)})
    roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
    threshold = roc_t['threshold']
    threshold = list(threshold)[0]
    
    y_preds = [1 if ele >= threshold else 0 for ele in predictions] 
    
    
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(true_label, predictions, average="binary", pos_label=0)
    #### conf_matrix = [["true_normal", "false_abnormal"], ["false_normal", "true_abnormal"]]     
    conf_matrix = metrics.confusion_matrix(true_label, predictions)
    performance = OrderedDict([ ('AUC', roc_score), ('precision', precision),
                                ("recall", recall), ("F1_Score", f1_score), ("conf_matrix", conf_matrix),
                                ("threshold", threshold)])
                                
    print(performance)
    

