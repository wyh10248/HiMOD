import numpy as np
import pandas as pd

# 示例矩阵
matrix1 = np.loadtxt('D:/桌面/MDA/dataset/DrugVirus/GSD.csv',delimiter=',')
matrix2 = np.loadtxt('D:/桌面/MDA/dataset/DrugVirus/drugfeatures.txt')

# 先求平均
fusion_matrix = (matrix1 + matrix2) / 2

# 找出 matrix1 中值为 0 的位置
zero_mask = (matrix1 == 0)

# 替换对应位置的值为 matrix2 的值
fusion_matrix[zero_mask] = matrix2[zero_mask]

print(fusion_matrix)
pd.DataFrame(fusion_matrix).to_csv('DSM1.csv', index=False, header=False)