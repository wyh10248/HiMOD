import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import KFold
from clac_metric import get_metrics
from sklearn.metrics import auc, roc_curve, precision_recall_curve
import random
from utils import *
from model import *
import gc
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
import warnings
import configparser
import os
from pathlib import Path
import pickle
warnings.filterwarnings("ignore")

dataset = 'MDAD' 
#dataset = 'DrugVirus'   
#dataset = 'aBiofilm'  #microbe 140  drug 1720
ini_path = os.path.join('..', 'config', f'{dataset}.ini')

# 读取配置
config = configparser.ConfigParser()
config.optionxform = str  
config.read(ini_path)
dataset_config = config[dataset]

parser = argparse.ArgumentParser()
for key, value in dataset_config.items():
    try:
        if '.' in value:
            value = float(value)
        else:
            value = int(value)
    except ValueError:
        pass  
    parser.add_argument(f'--{key}', default=value)

args = parser.parse_args()

device = torch.device ("cpu")

def load_fold_data(data_dir, fold_k):
    
    save_path = Path(data_dir)
    fold_file = save_path / f'fold_{fold_k}.pkl'
    
    with open(fold_file, 'rb') as f:
        fold_data = pickle.load(f)
    
        
    return fold_data

def cross_validation_experiment(args, data_dir='../dataset/MDAD'):
    metric = np.zeros((1, 7))
    score =[]
    tprs=[]
    fprs=[]
    aucs=[]
    precisions=[]
    recalls = []
    auprs = []
    outputs_all=[]
    pre_matrix = np.zeros((args.microbe_len, args.dis_len))
    print("seed=%d, evaluating drug-microbe...." % (args.seed))
    for k in range(args.k_fold):   
        print("------this is %dth cross validation------" % (k + 1))
        fold_data = load_fold_data(data_dir, k)
        train_matrix = fold_data['train_matrix']
        feature_matrix = fold_data['x']
        edge_index = fold_data['edge_index']
        edge_attr = fold_data['edge_attr']
        A = fold_data['A'] 
        test_index = fold_data['test_index'] 
        microbe_len = A.shape[0]
        dis_len = A.shape[1]
        #print(test_index)
        
        edge_index = edge_index.to(torch.int64)
        edge_attr = edge_attr.to(torch.float32)
        
        model = hu(args)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

        #训练循环
        model.train()
        for epoch in range(args.epoch):
            optimizer.zero_grad()
            pred, label = model(args, feature_matrix, train_matrix, edge_attr, edge_index, train_model=True) 
            loss = F.binary_cross_entropy(pred.float(), label)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

          # 模型评估
        model.eval()
        with torch.no_grad():
            pred_scores, _, outputs = model(args, feature_matrix, A, edge_attr, edge_index, train_model=False)

       
        micro_dis_res= pred_scores.detach().cpu()
        predict_y_proba = micro_dis_res.reshape(microbe_len, dis_len).detach().numpy()
        pre_matrix[tuple(np.array(test_index).T)] = predict_y_proba[tuple(np.array(test_index).T)]  
        A = np.array(A)
        
        metric_tmp = get_metrics(A[tuple(np.array(test_index).T)],
                                  predict_y_proba[tuple(np.array(test_index).T)])
        fpr, tpr, t = roc_curve(A[tuple(np.array(test_index).T)],
                                  predict_y_proba[tuple(np.array(test_index).T)])
        precision, recall, _ = precision_recall_curve(A[tuple(np.array(test_index).T)],
                                  predict_y_proba[tuple(np.array(test_index).T)])
        tprs.append(tpr)
        fprs.append(fpr)
        precisions.append(precision)
        recalls.append(recall)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        auprs.append(metric_tmp[1])


        print(metric_tmp)
        metric += metric_tmp      #  五折交叉验证的结果求和
        score.append(pre_matrix)
        outputs_all.append(outputs)
        del train_matrix  # del只删除变量，不删除数据
        gc.collect()  # 垃圾回收
    print('Mean:', metric / args.k_fold)
    metric = np.array(metric / args.k_fold)   #  五折交叉验证的结果求均值
    return outputs_all, metric, score
        

def main(args):

    outputs_all, result, score = cross_validation_experiment(args)
    
if __name__== '__main__':
    main(args)


 