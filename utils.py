from itertools import chain
import json
import os

import cv2
from matplotlib import pyplot as plt
from visualizer import Visualizer
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision

import time
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support, confusion_matrix, average_precision_score, fbeta_score
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from mpl_toolkits.axes_grid1 import make_axes_locatable
class Utils():

    #--------------------------------------------------------------------------------#
    def __init__(self, batch_size, device='cuda', cfg=None):
    # Parameters:
        self.batch_size   = batch_size
        self.device       = device
        self.visualizer = Visualizer(cfg, utils=self)
        self.cfg = cfg
        
    #--------------------------------------------------------------------------------#
    def getCrossEntropyLoss(self, binary=False):
        #print("[Using CrossEntropyLoss...]")
        if binary:
            criterion = nn.BCELoss()
        else:
            criterion = nn.CrossEntropyLoss()

        return (criterion)

    #--------------------------------------------------------------------------------#
    def getSGDOptimizer(self, model, learningRate = 0.001, momentum=0.9):
        #print("[Using small learning rate with momentum...]")
        optimizer_conv = optim.SGD(list(filter(
            lambda p: p.requires_grad, model.parameters())), lr=learningRate, momentum=momentum)

        return (optimizer_conv)

    #--------------------------------------------------------------------------------#
    def getLrScheduler(self, model, step_size=7, gamma=0.1):
        print("[Creating Learning rate scheduler...]")
        exp_lr_scheduler = lr_scheduler.StepLR(model, step_size=step_size, gamma=gamma)

        return (exp_lr_scheduler)

    #--------------------------------------------------------------------------------#
    ''' Train function '''
    def train(self, model, dataloader, criterion, optimizer, epoch=None):
        model.train()
        total = 0
        lossTotal = 0
        y_preds = [] # Store all predictions, for metric analysis
        y_trues = []

        for i, data in enumerate(tqdm(dataloader)):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, _ = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            #print(f'Input shape: {outputs.shape}')
            #print(f'Layer: {layer}')
            output = model(inputs)

            #_, predicted = torch.max(outputs.data, 1)
            loss = criterion(output, labels.float().unsqueeze(1))
            loss.backward()
            
            if self.cfg.display:
                self.visualizer.plot_current_errors(total_steps=i*(epoch+1), errors=loss.item())
            
            optimizer.step()

            # print statistics
            running_loss = loss.item() * inputs.size(0)
            lossTotal += running_loss # epoch loss
            y_preds +=  list(output.detach().cpu().numpy())
            y_trues += list(labels.cpu().numpy())

            #if i % 200 == 0:    # print every 200 mini-batches
            #    print('[%d, %5d] loss: %.5f ; Accuracy: %.2f'%
            #        (epoch, i + 1, running_loss/total_batch, 100 * correct_batch / total_batch))
            
            running_loss = 0.0
            total += labels.size(0)
            
        performance, t, y_preds_man, y_preds_auc = self.get_performance(y_preds=y_preds, y_trues=y_trues)
        print("Performance", str(performance))
        if self.cfg.display:
            self.visualizer.plot_performance(epoch=epoch, performance=performance, tag="Train_Performance_AutoClassifier")
            self.visualizer.plot_current_conf_matrix(epoch=epoch, cm=performance["conf_matrix"], tag="Train_Confusion_Matrix_AutoClassifier")
            self.visualizer.plot_pr_curve(y_preds=y_preds, y_trues=y_trues, t=t, global_step=epoch, tag="Train_PR_Curve_AutoClassifier")
        if epoch != None:
            print(f'Epoch {epoch} - Train Performance: {performance}, Loss: {lossTotal}')

        return model, performance, lossTotal, y_preds

    #--------------------------------------------------------------------------------#
    ''' Test function '''
    def test(self, model, testloader, criterion, epoch=None, outf=None, network=None):
        model.eval()
        total = 0
        running_loss = 0
        y_preds = [] # Store all predictions, for metric analysis
        y_trues = []

        with torch.no_grad():
            for data in tqdm(testloader):
                inputs, labels, _ = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                #outputs = model(inputs)

                output = model(inputs)

                loss = criterion(output, labels.float().unsqueeze(1))
                running_loss += loss.item() * inputs.size(0)

                #_, predicted = torch.max(output.data, 1)

                total += labels.size(0)
                #correct += (predicted == labels).sum().item()
                y_preds +=  list(output.detach().cpu().numpy())
                y_trues += list(labels.cpu().numpy())

        #acc  = 100 * correct / total
        lossTotal = running_loss / total
        performance, t, y_preds_man, y_preds_auc = self.get_performance(y_preds=y_preds, y_trues=y_trues)
        with open(os.path.join(outf, str(self.cfg.name) + "_" + network +".txt"), "a") as f:
            f.write(f'Epoch {epoch} - Val Performance: {str(performance)},    Loss: {str(loss)}')
            f.close()
        if self.cfg.display:
            self.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["threshold"], global_step=epoch, save_path=os.path.join(outf,"histogram_False_"+str(epoch)+".png"), tag="Histogram_Test_"+str(self.cfg.name))
            self.visualizer.plot_performance(epoch=epoch, performance=performance, tag="Test_Performance_AutoClassifier")
            self.visualizer.plot_current_conf_matrix(epoch=epoch, cm=performance["conf_matrix"], save_path=os.path.join(outf, "conf_matrix_False_"+str(epoch)+".png"), tag="Test_Confusion_Matrix_AutoClassifier")
            self.visualizer.plot_pr_curve(y_preds=y_preds, y_trues=y_trues, t=t, global_step=epoch, tag="Test_PR_Curve_AutoClassifier")
            self.visualizer.plot_roc_curve(y_trues=y_trues, y_preds=y_preds, global_step=1, tag="ROC_Curve_Test", save_path=os.path.join(outf, "roc_False_"+str(epoch)+".png"))
        if epoch != None:
            print(f'Epoch {epoch} - Val Performance: {performance},    Loss: {loss}')
        return performance, lossTotal, y_preds

    #--------------------------------------------------------------------------------#
    ''' Inference function '''
    def inference(self, model, inferenceloader, network=None, outf=None):
        model.eval()
        y_preds = [] # Store all predictions, for metric analysis
        y_trues = []
        inferenceTime = []
        file_names = []
        #with torch.no_grad():
        for data in tqdm(inferenceloader):
            startTime = time.time()
            inputs, labels, file_name_batch = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            output = model(inputs)
            if self.cfg.save_anomaly_map:
                save_dir = os.path.join(self.cfg.outf, "ano_maps")
                if not os.path.isdir(save_dir): os.mkdir(save_dir)
                self.get_cam_of_model(model, model.cam_layer, inputs, save_dir, file_name_batch)
            y_preds.append(output.detach().cpu().squeeze().item())
            y_trues.append(labels.cpu().squeeze().item())

            inferenceTime.append(time.time()-startTime)
            file_names.extend(file_name_batch)
        performance, t, y_preds_man, y_preds_auc = self.get_performance(y_preds=y_preds, y_trues=y_trues, manual_threshold=self.cfg.decision_threshold)
        if self.cfg.display:
            self.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["threshold"], global_step=1, save_path=os.path.join(outf,"histogram_True.png"), tag="Histogram_Inference")
            self.visualizer.plot_performance(epoch=1, performance=performance, tag="Inference_Performance_AutoClassifier")
            self.visualizer.plot_current_conf_matrix(1, performance["conf_matrix"], save_path=os.path.join(outf, "conf_matrix_True.png"))
            if self.cfg.decision_threshold:
                self.visualizer.plot_current_conf_matrix(2, performance["conf_matrix_man"], save_path=os.path.join(outf, "conf_matrix_manTrue.png"))
                self.visualizer.plot_histogram(y_trues=y_trues, y_preds=y_preds, threshold=performance["manual_threshold"], global_step=2, save_path=os.path.join(outf, "histogram_manTrue.png"), tag="Histogram_Inference_man")
            self.visualizer.plot_roc_curve(y_trues=y_trues, y_preds=y_preds, global_step=1, tag="ROC_Curve", save_path=os.path.join(outf, "roc_True.png"))
            

            self.write_inference_result(file_names=file_names, y_preds=y_preds_auc, y_trues=y_trues, outf=os.path.join(outf,"classification_result_" + str(self.cfg.name) + "_" + network + ".json"))
            if self.cfg.decision_threshold:
                self.write_inference_result(file_names=file_names, y_preds=y_preds_man, y_trues=y_trues, outf=os.path.join(outf,"classification_result_" + str(self.cfg.name) + "_" + network + "_man.json"))
            
        with open(os.path.join(outf, str(self.cfg.name) + "_" + network +".txt"), "a") as f:
            f.write(f'Inf Performance: {str(performance)}, Inf_times: {str(sum(inferenceTime))}')
            f.close()
        print(f'Inf Performance: {performance}')
        print (f'Inference time: {sum(inferenceTime)} secs')
        print (f'Inference time / individual: {sum(inferenceTime)/len(y_trues)} secs')
        return file_names, y_preds_auc, y_trues


    #--------------------------------------------------------------------------------#
    def initializeModel(self, model_name, num_classes, use_pretrained=True, input_size=256):
        model_name = model_name.lower()
        model = None
        input_size = input_size

        if "resnet" in model_name:
            """ Resnet18
            """
            if '18' in model_name:
                model = torchvision.models.resnet18(pretrained=use_pretrained)
            elif '34' in model_name:
                model = torchvision.models.resnet34(pretrained=use_pretrained)
            elif '50' in model_name:
                model = torchvision.models.resnet50(pretrained=use_pretrained)
            elif '101' in model_name:
                model = torchvision.models.resnet101(pretrained=use_pretrained)
            elif '152' in model_name:
                model = torchvision.models.resnet152(pretrained=use_pretrained)
            num_ftrs = model.fc.in_features
            model.cam_layer = model.layer4[-1]
            model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())

        elif model_name == "alexnet":
            """ Alexnet
            """
            model = torchvision.models.alexnet(pretrained=use_pretrained)
            num_ftrs = model.classifier[6].in_features
            model.cam_layer = model.features[-1]
            model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())

        elif "vgg" in model_name:
            """ VGG
            """
            if '11' in model_name:
                if 'bn' in model_name:
                    model = torchvision.models.vgg11_bn(pretrained=use_pretrained)
                else:
                    model = torchvision.models.vgg11(pretrained=use_pretrained)
            elif '13' in model_name:
                if 'bn' in model_name:
                    model = torchvision.models.vgg13_bn(pretrained=use_pretrained)
                else:
                    model = torchvision.models.vgg13(pretrained=use_pretrained)
            elif '16' in model_name: 
                if 'bn' in model_name:
                    model = torchvision.models.vgg16_bn(pretrained=use_pretrained)
                else:
                    model = torchvision.models.vgg16(pretrained=use_pretrained)
            elif '19' in model_name:
                if 'bn' in model_name:
                    model = torchvision.models.vgg19_bn(pretrained=use_pretrained)
                else:
                    model = torchvision.models.vgg19(pretrained=use_pretrained)
            else:
                print("Invalid model name, returning 'None'...")
                return None

            num_ftrs = model.classifier[6].in_features
            model.cam_layer = model.features[-1]
            model.classifier[6] = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())

        elif "squeezenet" in model_name:
            """ Squeezenet
            """
            if '1_1' in model_name:
                model = torchvision.models.squeezenet1_1(pretrained=use_pretrained)
            else:
                model = torchvision.models.squeezenet1_0(pretrained=use_pretrained)
            num_ftrs = model.classifier[1].in_channels
            model.classifier[1] = nn.Sequential(nn.Conv2d(num_ftrs, num_classes, kernel_size=(1,1), stride=(1,1)), nn.Sigmoid())
            model.cam_layer = model.features[-1]
            model.num_classes = num_classes

        elif "densenet" in model_name:
            """ Densenet
            """
            if '161' in model_name:
                model = torchvision.models.densenet161(pretrained=use_pretrained)
            if '169' in model_name:
                model = torchvision.models.densenet169(pretrained=use_pretrained)
            if '201' in model_name:
                model = torchvision.models.densenet201(pretrained=use_pretrained)
            else: # else, get densenet121
                model = torchvision.models.densenet121(pretrained=use_pretrained)

            num_ftrs = model.classifier.in_features
            model.cam_layer = model.features[-1]
            model.classifier = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model = torchvision.models.inception_v3(pretrained=use_pretrained)
            # Handle the auxilary net
            num_ftrs = model.AuxLogits.fc.in_features
            model.AuxLogits.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
            # Handle the primary net
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        
        elif "efficientnet" in model_name:
            model = EfficientNet.from_pretrained(model_name)
            num_ftrs = model._fc.in_features
            model._fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        else:
            print("Invalid model name, returning 'None'...")
            return None

        return model, input_size
    
    def get_performance(self, y_trues, y_preds, manual_threshold=None):
        fpr, tpr, t = roc_curve(y_trues, y_preds)
        roc_score = auc(fpr, tpr)
        ap = average_precision_score(y_trues, y_preds, pos_label=1)
        recall_dict = dict()
        precisions = [0.996, 0.99, 0.95, 0.9]
        temp_dict=dict()
        min_thresh = 0.9*min(y_preds)
        max_thresh = 1.1*max(y_preds)
        print(max_thresh)
        mov_thresh = np.random.default_rng().uniform(min_thresh, max_thresh, 400)
        print(mov_thresh.shape)
        for th in sorted(mov_thresh, reverse=True):
            y_preds_new = [1 if ele >= th else 0 for ele in y_preds] 
            if len(set(y_preds_new)) == 1:
                print("y_preds_new did only contain the element {}... Continuing with next iteration!".format(y_preds_new[0]))
                continue
            
            precision, recall, _, _ = precision_recall_fscore_support(y_trues, y_preds_new, average="binary", pos_label=1)
            temp_dict[str(precision)] = recall
        p_dict = OrderedDict(sorted(temp_dict.items(), reverse=False))
        # interploation
        print("interpolation steps", len(list(p_dict.keys())))
        for i in range(len(list(p_dict.keys())), 1, -1):
            if p_dict[list(p_dict.keys())[i-1]]>p_dict[list(p_dict.keys())[i-2]]:
                p_dict[list(p_dict.keys())[i-2]] = p_dict[list(p_dict.keys())[i-1]]
        print("finished interpolation")
        p_dict = OrderedDict(sorted(p_dict.items(), reverse=True))
        for p in precisions:   
            recall_dict["recall at pr="+str(p)] = 0.0
            recall_dict["true pr="+str(p)] = 0.0
            for precision, recall in p_dict.items(): 
                if float(precision)>=0.998*p:
                    recall_dict["recall at pr="+str(p)] = recall
                    recall_dict["true pr="+str(p)] = float(precision)
                    continue
                else:
                    break

        
        # auroc Threshold
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(t, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
        auc_threshold = roc_t['threshold']
        auc_threshold = list(auc_threshold)[0]
        
        
        
        y_preds_auc_thresh = [1 if ele >= auc_threshold else 0 for ele in y_preds] 
        
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_trues, y_preds_auc_thresh, average="binary", pos_label=1)
        f05_score = fbeta_score(y_trues, y_preds_auc_thresh, beta=0.5, average="binary", pos_label=1)
        #### conf_matrix = [["true_normal", "false_abnormal"], ["false_normal", "true_abnormal"]]     
        conf_matrix = confusion_matrix(y_trues, y_preds_auc_thresh)
        performance = OrderedDict([ ('auc', roc_score), ("ap", ap), ('precision', precision),
                                    ("recall", recall), ("f1_score", f1_score), ("f05_score", f05_score), ("conf_matrix", conf_matrix),
                                    ("threshold", auc_threshold)])
        
        if manual_threshold:
            man_dict = dict()
            y_preds_man_thresh = [1 if ele >= manual_threshold else 0 for ele in y_preds]
            precision_man, recall_man, f1_score_man, _ = precision_recall_fscore_support(y_trues, y_preds_man_thresh, average="binary", pos_label=1)
            f05_score_man = fbeta_score(y_trues, y_preds_man_thresh, beta=0.5, average="binary", pos_label=1)
            #### conf_matrix = [["true_normal", "false_abnormal"], ["false_normal", "true_abnormal"]]     
            conf_matrix_man = confusion_matrix(y_trues, y_preds_man_thresh)
            man_dict["manual_threshold"] = manual_threshold
            man_dict["precision_man"] = precision_man
            man_dict["recall_man"] = recall_man
            man_dict["f1_score_man"] = f1_score_man
            man_dict["f05_score_man"] = f05_score_man
            man_dict["conf_matrix_man"] = conf_matrix_man
            performance.update(man_dict)
        performance.update(recall_dict)
                                    
        return performance, t, y_preds_man_thresh if manual_threshold else None, y_preds_auc_thresh

    def get_values_for_pr_curve(self, y_trues, y_preds, thresholds):
        precisions = []
        recalls = []
        tn_counts = []
        fp_counts = []
        fn_counts = []
        tp_counts = []
        for threshold in thresholds:
            y_preds_new = [1 if ele >= threshold else 0 for ele in y_preds] 
            tn, fp, fn, tp = confusion_matrix(y_trues, y_preds_new).ravel()
            if len(set(y_preds_new)) == 1:
                print("y_preds_new did only contain the element {}... Continuing with next iteration!".format(y_preds_new[0]))
                continue
            
            precision, recall, _, _ = precision_recall_fscore_support(y_trues, y_preds_new, average="binary", pos_label=1)
            precisions.append(precision)
            recalls.append(recall)
            tn_counts.append(tn)
            fp_counts.append(fp)
            fn_counts.append(fn)
            tp_counts.append(tp)
            
            
        
        return np.array(tp_counts), np.array(fp_counts), np.array(tn_counts), np.array(fn_counts), np.array(precisions), np.array(recalls), len(thresholds)
    
    def get_values_for_roc_curve(self, y_trues, y_preds):
        fpr, tpr, _ = roc_curve(y_trues, y_preds) 
        return fpr, tpr, auc(fpr, tpr)

    def write_inference_result(self, file_names, y_preds, y_trues, outf):
        #if any(isinstance(i, list) for i in y_trues):
        #    y_trues = list(chain(*y_trues))
        #if any(isinstance(i, list) for i in y_preds):
         #   y_preds = list(chain(*y_preds))
        classification_result = {"tp": [], "fp": [], "tn": [], "fn": []}
        for file_name, gt, anomaly_score in zip(file_names, y_trues, y_preds):
            anomaly_score=int(anomaly_score)
            if gt == anomaly_score == 0:
                classification_result["tp"].append(file_name)
            if anomaly_score == 0 and gt != anomaly_score:
                classification_result["fp"].append(file_name)
            if gt == anomaly_score == 1:
                classification_result["tn"].append(file_name)
            if anomaly_score == 1 and gt != anomaly_score:
                classification_result["fn"].append(file_name)
                    
        with open(outf, "w") as file:
            json.dump(classification_result, file, indent=4)
    # using pytorch-grad-cam (https://github.com/jacobgil/pytorch-grad-cam)
    def get_cam_of_model(self, model, target_layers, input_tensor, save_dir, file_names):
        start_time = time.time()
        cam = GradCAM(model=model, target_layer=target_layers, use_cuda=True if self.device == "cuda" else False)
        amaps=cam(input_tensor=input_tensor)
        amaps = np.nan_to_num(amaps)# for some reason, some amaps returned from grad_cam are nan... setting to zero
        assert amaps.shape[0] == input_tensor.shape[0]
        assert amaps.shape[0] == len(file_names)    # TODO: for some reason i implemented file_paths as tuples... maybe change this later
        for i in range(amaps.shape[0]):
            save_path = os.path.join(save_dir, file_names[i])
            fig, axis = plt.subplots(figsize=(4,4))
            axis.imshow(input_tensor[i, :].permute(1, 2, 0).cpu().numpy())
            divider = make_axes_locatable(axis)
            cax = divider.append_axes('right', size='5%', pad=0.1)
            amap_on_image = axis.imshow(amaps[i, :], alpha=.7, cmap='viridis', vmin=0, vmax=1)
            fig.colorbar(amap_on_image, cax=cax, orientation='vertical')
            fig.savefig(save_path)
            plt.close()
    