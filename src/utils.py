import csv
import numpy as np
import matplotlib.pyplot as plt
from torch.utils import data
import random
import torch

device = torch.device("cpu")

def set_seed(seed=50):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
def find_one_zero(RS):
    one_postive = np.argwhere(RS == 1)
    one_postive = np.array(one_postive)
    np.random.shuffle(one_postive)

    zero_postive = np.argwhere(RS == 0)
    zero_postive = np.array(zero_postive)
    np.random.shuffle(zero_postive)  

    return one_postive, zero_postive

def get_node_types(mirna_disease_matrix):
    num_miRNA = mirna_disease_matrix.shape[0]
    num_disease = mirna_disease_matrix.shape[1]

    node_types = torch.tensor(
        [0] * num_miRNA +   # miRNA
        [1] * num_disease   # disease
    )
    return node_types

def index_dp(A, k):
    index_drug_p = np.transpose(np.nonzero(A))
    np.random.shuffle(index_drug_p)
    data_1 = np.array_split(index_drug_p, k, 0)

    index_dp_zero = np.argwhere(A == 0)
    np.random.shuffle(index_dp_zero)
    data_0 = np.array_split(index_dp_zero, k, 0)

    return data_0, data_1

def Split_data(data_1, data_0, fold, k, drug_p):
    X_train = []  
    X_test = []

    for i in range(k):  
        if i != fold:  # 如果不是当前折，则用作训练集
            for j in range(len(data_1[i])):
                X_train.append(data_1[i][j])
            for t in range(len(data_0[i])):
                if t < len(data_1[i]):  # 平衡正负样本数量
                    X_train.append(data_0[i][t])
                else:
                    x = int(data_0[i][t][0])
                    y = int(data_0[i][t][1])
                    X_test.append([x, y])
        else:  # 当前折用作测试集
            for t1 in range(len(data_1[i])):
                x = int(data_1[i][t1][0])  
                y = int(data_1[i][t1][1])  
                X_test.append([x, y])
                    
            for t2 in range(len(data_0[i])):  
                x = int(data_0[i][t2][0])  
                y = int(data_0[i][t2][1])
                X_test.append([x, y])

    np.random.shuffle(X_train)
    return X_train, X_test

def Preproces_Data(RS, test_id):
    copy_RS = RS / 1
    for i in range(test_id.shape[0]):
        x = int(test_id[i][0])
        y = int(test_id[i][1])
        copy_RS[x, y] = 0
    return copy_RS

def load_data(id, RS):
    import torch.utils.data as Data
    x = []
    y = []
    for j in range(id.shape[0]):  
        temp_save = []
        x_A = int(id[j][0])  
        y_A = int(id[j][1])  
        temp_save.append([x_A, y_A])  
        
        label = RS[x_A, y_A]
        
        x.append(temp_save)
        y.append(label)

    x = torch.FloatTensor(np.array(x))   # shape: (N, 1, 2)
    y = torch.LongTensor(np.array(y))   # shape: (N,)

    torch_dataset = Data.TensorDataset(x, y)
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=len(x),     # 一次加载所有数据
        shuffle=False,
        num_workers=0,
        drop_last=False
    )  
    return x, y, data2_loader
def constructNet(mirna_disease_matrix):
    drug_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[0], mirna_disease_matrix.shape[0]), dtype=np.int8))
    dis_matrix = np.matrix(
        np.zeros((mirna_disease_matrix.shape[1], mirna_disease_matrix.shape[1]), dtype=np.int8))

    mat1 = np.hstack((drug_matrix, mirna_disease_matrix))#沿着水平方向进行连接
    mat2 = np.hstack((mirna_disease_matrix.T, dis_matrix))
    adj = np.vstack((mat1, mat2))#沿着垂直方向进行连接
    return adj

def constructHNet(train_mirna_disease_matrix, mirna_matrix, disease_matrix):
    mat1 = np.hstack((mirna_matrix, train_mirna_disease_matrix))
    mat2 = np.hstack((train_mirna_disease_matrix.T, disease_matrix))
    node_embeddings = np.vstack((mat1, mat2))
    return node_embeddings

def train_features_choose(rel_adj_mat, features_embedding):
    rna_nums = rel_adj_mat.shape[0]
    features_embedding_rna = features_embedding[0:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:features_embedding.size()[0], :]
    train_features_input, train_lable = [], []
    # positive position index
    #rel_adj_mat = torch.from_numpy(rel_adj_mat)
    positive_index_tuple = torch.where(rel_adj_mat == 1)
    positive_index_list = list(zip(positive_index_tuple[0], positive_index_tuple[1]))

    for (r, d) in positive_index_list:
        # positive samples
        train_features_input.append((features_embedding_rna[r, :] * features_embedding_dis[d, :]).unsqueeze(0))
        train_lable.append(1)
        # negative samples
        negative_colindex_list = []
        for i in range(1):
            j = np.random.randint(rel_adj_mat.size()[1])
            while (r, j) in positive_index_list:
                j = np.random.randint(rel_adj_mat.size()[1])
            negative_colindex_list.append(j)
        for nums_1 in range(len(negative_colindex_list)):
            train_features_input.append(
                (features_embedding_rna[r, :] * features_embedding_dis[negative_colindex_list[nums_1], :]).unsqueeze(0))
        for nums_2 in range(len(negative_colindex_list)):
            train_lable.append(0)
    train_features_input = torch.cat(train_features_input, dim=0)
    train_lable = torch.FloatTensor(np.array(train_lable)).unsqueeze(1)
    return train_features_input.to(device), train_lable.to(device)

def test_features_choose(rel_adj_mat, features_embedding):
    rna_nums, dis_nums = rel_adj_mat.shape[0], rel_adj_mat.shape[1]
    features_embedding_rna = features_embedding[0:rna_nums, :]
    features_embedding_dis = features_embedding[rna_nums:features_embedding.size()[0], :]
    test_features_input, test_lable = [], []

    for i in range(rna_nums):
        for j in range(dis_nums):
            test_features_input.append((features_embedding_rna[i, :] * features_embedding_dis[j, :]).unsqueeze(0))
            test_lable.append(rel_adj_mat[i, j])

    test_features_input = torch.cat(test_features_input, dim=0)
    
    return test_features_input.to(torch.float32),test_lable

def sort_matrix(score_matrix, interact_matrix):
    '''
    实现矩阵的列元素从大到小排序
    1、np.argsort(data,axis=0)表示按列从小到大排序
    2、np.argsort(data,axis=1)表示按行从小到大排序
    '''
    sort_index = np.argsort(-score_matrix, axis=0)  # 沿着行向下(每列)的元素进行排序
    score_sorted = np.zeros(score_matrix.shape)
    y_sorted = np.zeros(interact_matrix.shape)
    for i in range(interact_matrix.shape[1]):
        score_sorted[:, i] = score_matrix[:, i][sort_index[:, i]]
        y_sorted[:, i] = interact_matrix[:, i][sort_index[:, i]]
    return y_sorted, score_sorted




#计算邻接矩阵
def get_adjacency_matrix(similarity_matrix, threshold):
    n = similarity_matrix.shape[0]
    adjacency_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i][j] >= threshold:
                adjacency_matrix[i][j] = 1
                adjacency_matrix[j][i] = 1

    return adjacency_matrix


def adjacency_matrix_to_edge_index(adjacency_matrix):
    adjacency_matrix = torch.tensor(adjacency_matrix)
    edge_index = torch.nonzero(adjacency_matrix, as_tuple=False).t()  # (2, E)
    edge_attr = adjacency_matrix[edge_index[0], edge_index[1]].unsqueeze(1)  # (E, 1) 边权作为边特征
    return edge_index, edge_attr



def transfer_label_from_prob(proba, threshold):
    proba = (proba - proba.min()) / (
            proba.max() - proba.min())
    label = [1 if val >= threshold else 0 for val in proba]
    return label

def balance_samples(pos_index, neg_index):

    pos_num = len(pos_index)
    neg_num = len(neg_index)
    if pos_num > neg_num:
        # 对正样本进行下采样
        balanced_pos_index = random.sample(pos_index, neg_num)
        balanced_neg_index = neg_index
    else:
        # 对负样本进行下采样
        balanced_pos_index = pos_index
        balanced_neg_index = random.sample(neg_index, pos_num)
    return balanced_pos_index, balanced_neg_index
def random_index(index_matrix, sizes):
    set_seed(sizes.seed)
    association_nam = index_matrix.shape[1]
    random_index = index_matrix.T.tolist()
    random.seed(sizes.seed)  # 获得相同随机数
    random.shuffle(random_index)  # 将原列表的次序打乱
    k_folds = sizes.k_fold
    CV_size = int(association_nam / k_folds)
    temp = np.array(random_index[:association_nam - association_nam %
                                  k_folds]).reshape(k_folds, CV_size, -1).tolist()
    temp[k_folds - 1] = temp[k_folds - 1] + \
                        random_index[association_nam - association_nam % k_folds:]
    return temp

class Sizes(object):
    def __init__(self, drug_size, mic_size):
        self.c = 12

'''def get_adjacency_matrix(feat, k):
    # compute C
    featprod = np.dot(feat.T, feat)
    smat = np.tile(np.diag(featprod), (feat.shape[1], 1))
    dmat = smat + smat.T - 2 * featprod
    dsort = np.argsort(dmat)[:, 1:k + 1]
    C = np.zeros((feat.shape[1], feat.shape[1]))
    for i in range(feat.shape[1]):
        for j in dsort[i]:
            C[i, j] = 1.0

    return C'''

def plot_auc_curves(fprs, tprs, aucs):
    mean_fpr = np.linspace(0, 1, 1000)
    tpr = []
    #plt.style.use('ggplot')
    for i in range(len(fprs)):
        tpr.append(np.interp(mean_fpr, fprs[i], tprs[i]))
        tpr[-1][0] = 0.0
        plt.plot(fprs[i], tprs[i], alpha=0.8, label='ROC fold %d (AUC = %.4f)' % (i + 1, aucs[i]))

    mean_tpr = np.mean(tpr, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    auc_std = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', alpha=0.8, label='Mean AUC (AUC = %.4f $\pm$ %.4f)' % (mean_auc, auc_std))
    plt.plot([-0.05, 1.05], [-0.05, 1.05], linestyle='--', color='navy', alpha=0.4)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.rcParams.update({'font.size': 10})
    plt.legend(loc='lower right', prop={"size": 8})
    #plt.savefig('./auc.jpg', dpi=1200, bbox_inches='tight') 
    plt.show()
    

def plot_prc_curves(precisions, recalls, auprs):
    mean_recall = np.linspace(0, 1, 1000)
    precision = []
    #plt.style.use('ggplot')
    for i in range(len(recalls)):
        precision.append(np.interp(1-mean_recall, 1-recalls[i], precisions[i]))
        precision[-1][0] = 1.0
        plt.plot(recalls[i], precisions[i], alpha=0.8, label='ROC fold %d (AUPR = %.4f)' % (i + 1, auprs[i]))

    mean_precision = np.mean(precision, axis=0)
    mean_precision[-1] = 0
    # mean_prc = metrics.auc(mean_recall, mean_precision)
    mean_prc = np.mean(auprs)
    prc_std = np.std(auprs)
    plt.plot(mean_recall, mean_precision, color='b', alpha=0.8,
             label='Mean AUPR (AUPR = %.4f $\pm$ %.4f)' % (mean_prc, prc_std))  # AP: Average Precision
    plt.plot([-0.05, 1.05], [1.05, -0.05], linestyle='--', color='navy', alpha=0.4)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('PR curve')
    plt.rcParams.update({'font.size': 10})
    plt.legend(loc='lower left', prop={"size": 8})
    #plt.savefig('./pr.jpg', dpi=1200, bbox_inches='tight') 
    plt.show()
