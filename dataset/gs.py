import numpy as np
import pandas as pd

def gsl(ldg_file):
    """
    计算 lncRNA 的高斯相似性矩阵。

    参数:
    ldg_file (str): 包含 lncRNA-disease关联矩阵的文件路径，文件应是一个二维矩阵。

    返回:
    lncrna_gs_sim (np.ndarray): 计算得到的 lncRNA 高斯相似性矩阵。
    """
    # 加载 lncRNA-疾病关联矩阵
    LDG = ldg_file  # 假设文件是文本格式的矩阵

    # 矩阵的行数 (lncRNA 的数量)
    nl = LDG.shape[0]

    # 计算归一化参数 normSum
    norm_sum = 0
    for i in range(nl):
        norm_sum += np.linalg.norm(LDG[i, :], 2) ** 2
    rl = 1 / (norm_sum / nl)

    # 初始化高斯相似性矩阵
    lncrna_gs_sim = np.zeros((nl, nl))

    # 计算高斯相似性
    for i in range(nl):
        for j in range(nl):
            sub = LDG[i, :] - LDG[j, :]
            lncrna_gs_sim[i, j] = np.exp(-rl * (np.linalg.norm(sub, 2) ** 2))


    return lncrna_gs_sim

if __name__ == "__main__":
    ldg_file = np.loadtxt('D:/Desktop文件/MDA/dataset/aBiofilm/drug_microbe_adjacency.csv', delimiter=',')
    ldg_file = ldg_file.T
    result = gsl(ldg_file)
    df = pd.DataFrame(result)
    df.to_csv('D:/Desktop文件/MDA/dataset/aBiofilm/GSM.csv', index=False, header=False)

