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

def crossval_index(drug_mic_matrix, sizes):
    random.seed(sizes.seed)
    set_seed(args.seed)
    pos_index_matrix = np.mat(np.where(drug_mic_matrix == 1))
    neg_index_matrix = np.mat(np.where(drug_mic_matrix == 0))
    pos_index = random_index(pos_index_matrix, sizes)
    neg_index = random_index(neg_index_matrix, sizes)
    index = []
    for i in range(sizes.k_fold):
        balanced_pos_index, balanced_neg_index = balance_samples(pos_index[i], neg_index[i])
        index.append(balanced_pos_index + balanced_neg_index)
    return index

def cross_validation_experiment(A, microSimi, disSimi, args):
    index = crossval_index(A, args)
    metric = np.zeros((1, 7))
    score =[]
    tprs=[]
    fprs=[]
    aucs=[]
    precisions=[]
    recalls = []
    auprs = []
    pre_matrix = np.zeros(A.shape)
    print("seed=%d, evaluating drug-microbe...." % (args.seed))
    for k in range(args.k_fold):
        print("------this is %dth cross validation------" % (k + 1))
        
        train_matrix = np.matrix(A, copy=True)
        train_matrix[tuple(np.array(index[k]).T)] = 0  # 将5折中的一折变为0
        x=constructHNet(train_matrix, microSimi, disSimi)
        edge_index, edge_attr =adjacency_matrix_to_edge_index(train_matrix)
        node_types = get_node_types(train_matrix)
        microbe_len = A.shape[0]
        dis_len = A.shape[1]
        
        #train_matrix = torch.from_numpy(train_matrix).to(torch.float32)
        x = torch.from_numpy(x).to(torch.float32)
        edge_index = edge_index.to(torch.int64)
        edge_attr = edge_attr.to(torch.float32)
        A1 = constructNet(train_matrix)
        
        model = hu(args)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

        #训练循环
        model.train()
        for epoch in range(args.epoch):
            optimizer.zero_grad()
            pred, label = model(args, x, A, edge_attr, edge_index, train_model=True) 
            loss = F.binary_cross_entropy(pred.float(), label)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch+1} | Loss: {loss.item():.4f}")

         # 模型评估
        model.eval()
        with torch.no_grad():
            pred_scores, _ = model(args, x, train_matrix, edge_attr, edge_index, train_model=False)

        micro_dis_res= pred_scores.detach().cpu()
        predict_y_proba = micro_dis_res.reshape(microbe_len, dis_len).detach().numpy()
        pre_matrix[tuple(np.array(index[k]).T)] = predict_y_proba[tuple(np.array(index[k]).T)]  #从预测分数矩阵中取出验证集的预测结果 只返回相应的预测分数
        A = np.array(A)
        metric_tmp = get_metrics(A[tuple(np.array(index[k]).T)],
                                  predict_y_proba[tuple(np.array(index[k]).T)]) #预测结果所得的评价指标
        fpr, tpr, t = roc_curve(A[tuple(np.array(index[k]).T)],
                                  predict_y_proba[tuple(np.array(index[k]).T)])
        precision, recall, _ = precision_recall_curve(A[tuple(np.array(index[k]).T)],
                                  predict_y_proba[tuple(np.array(index[k]).T)])
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
        del train_matrix  # del只删除变量，不删除数据
        gc.collect()  # 垃圾回收
    print('Mean:', metric / args.k_fold)
    metric = np.array(metric / args.k_fold)   #  五折交叉验证的结果求均值
    return metric, score, microbe_len, dis_len, tprs, fprs, aucs, precisions, recalls, auprs
        

def main(args):
# #------data1---------
    data_path = '../dataset/'
    data_set = 'MDAD/'

    A = np.loadtxt(data_path + data_set + 'drug_microbe_adjacency.csv',delimiter=',')
    DSM = np.loadtxt(data_path + data_set + 'DSM1.csv',delimiter=',') 
    MSM = np.loadtxt(data_path + data_set + 'MSM1.csv',delimiter=',')
#------data2---------
    # data_path = '../dataset/'
    # data_set = 'DrugVirus/'

    # A = np.loadtxt(data_path + data_set + 'drug_microbe_adjacency.csv',delimiter=',')
    # DSM = np.loadtxt(data_path + data_set + 'DSM1.csv',delimiter=',') 
    # MSM = np.loadtxt(data_path + data_set + 'MSM1.csv',delimiter=',')

#-----------------------------------------
# #------data3---------
    # data_path = '../dataset/'
    # data_set = 'aBiofilm/'

    # A = np.loadtxt(data_path + data_set + 'drug_microbe_adjacency.csv',delimiter=',')
    # DSM = np.loadtxt(data_path + data_set + 'DSM1.csv',delimiter=',') 
    # MSM = np.loadtxt(data_path + data_set + 'MSM1.csv',delimiter=',')
    
#-------------------------
    result, score, microbe_len, dis_len, tprs, fprs, aucs, precisions, recalls, auprs = cross_validation_experiment(A, DSM, MSM, args)
    #final_score = np.zeros_like(A, dtype=np.float32)
    #for fold_score in score:
    #    final_score += fold_score
    # 转换为 DataFrame（行列与原始矩阵一致）
    #df = pd.DataFrame(final_score)
    
    # 保存为 CSV 文件（不保存索引和列名）
    #output_filename = f"{dataset}_prediction_score.csv"
    #df.to_csv(output_filename, index=False, header=False)
    sizes = Sizes(microbe_len, dis_len)
    score_matrix = np.mean(score, axis=0)
    print(list(sizes.__dict__.values()) + result.tolist()[0][:2])
    plot_auc_curves(fprs, tprs, aucs)
    plot_prc_curves(precisions, recalls, auprs)
    


if __name__== '__main__':
    main(args)


 
