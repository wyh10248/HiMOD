import numpy as np
import pandas as pd


matrix1 = np.loadtxt('D:/桌面/MDA/dataset/DrugVirus/GSD.csv',delimiter=',')##
matrix2 = np.loadtxt('D:/桌面/MDA/dataset/DrugVirus/drugfeatures.txt')


fusion_matrix = (matrix1 + matrix2) / 2


zero_mask = (matrix1 == 0)


fusion_matrix[zero_mask] = matrix2[zero_mask]

print(fusion_matrix)

pd.DataFrame(fusion_matrix).to_csv('DSM1.csv', index=False, header=False)

