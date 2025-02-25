import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from func import *
from args import *
import openai
import pandas as pd
import math
from collections import Counter

parser = get_parser()
args = parser.parse_args()

print("111111")
with open("../data/cl100_embeddings.json", 'r') as f:
    cl100_emb = json.load(f)
    vector_data_json = {k: cl100_emb[k] for k in list(cl100_emb.keys())[:11000]} #词库总共有11000个词
    cl100_emb = None
    # 将这些键值对中的向量数据转换为 NumPy 数组，以方便进行数学和向量操作
    token_to_vector_dict = {token: np.array(vector) for token, vector in vector_data_json.items()}

# print(token_to_vector_dict)

if not os.path.exists(f'../data/sorted_cl100_embeddings.json'):
    init_func(args.eps_fx, token_to_vector_dict)
with open('../data/sorted_cl100_embeddings.json', 'r') as f1:
    sorted_cl100_emb = json.load(f1)
with open('../data/sensitivity_of_embeddings.json', 'r') as f:
    sen_emb = np.array(json.load(f))
print("222")


# 1. 使用最近邻得到每个词的第 k 个最近邻的距离，平均距离为阈值Y
def calculate_threshold_Y(k):
    Y=0
    for key in sorted_cl100_emb.keys():
        Y=Y+sorted_cl100_emb[key][k][1]
    Y=Y/len(sorted_cl100_emb.keys())
    return Y

# 2. 计算某个词在阈值 Y 范围内的邻接词数量
def count_neighbors_within_threshold(Y,word):
    distance = Y
    sorted_distances_for_token = sorted_cl100_emb.get(word, None) # 从 sorted_distance_data 中获取 origin_token 对应的距离信息列表
    distances_only = np.array([item[1] for item in sorted_distances_for_token]) # 提取 sorted_distances_for_token 列表中的距离部分，存储到 NumPy 数组
    index = np.searchsorted(distances_only, distance) # 找到 distance 在 distances_only 数组中的插入位置 index，保持数组的升序顺序
    neighbor_counts= int(index) # 距离小于 Y 的邻居数量，包括自身
    return neighbor_counts


# 3. 绘制直方图
def plot_histogram(K_values,Y_values):
    plt.figure(figsize=(10, 6))

    #统计每个词在阈值 Y 范围内的邻接词数量
    word_list = list(token_to_vector_dict.keys())  # 包含所有词汇
    neighbor_counts={}
    for word in word_list:
        neighbor_counts[word] = count_neighbors_within_threshold(Y_values,word)

    # 获取所有邻接词数量的值
    neighbor_values = list(neighbor_counts.values())
    # 统计每个邻接词数量出现的次数
    count_of_neighbors = Counter(neighbor_values)
    
    x = list(count_of_neighbors.keys())
    y = list(count_of_neighbors.values())

    # 绘制直方图
    plt.bar(x, y)
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Number of Tokens')
    plt.title(f'Histogram of Neighbor Counts per Token (Y={Y_values:.4f})')

    plt.grid()

    plt.savefig(f'fig/K{K_values}.png')

# k 的初始值（可调节）
K=[20]
Y=list(range(len(K)))

print("333")
# 4. 选择多个阈值 Y 来进行对比绘制
for i in range(len(K)):
    Y[i]= calculate_threshold_Y(K[i])
    print(f"Calculated threshold Y_{K[i]}: {Y[i]}")
    plot_histogram(K[i],Y[i])

print("444")


