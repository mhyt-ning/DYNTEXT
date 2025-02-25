from func import *
from args import *
import openai
import pandas as pd
import math
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np

parser = get_parser()
args = parser.parse_args()

os.makedirs(f"data/eps_fx_{args.eps_fx}_K{args.K}", exist_ok=True)

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

delta = 1e-5       # 置信度参数

#计算阈值Y
Y=0
for key in sorted_cl100_emb.keys():
    Y=Y+sorted_cl100_emb[key][args.K][1]

Y=Y/len(sorted_cl100_emb.keys())
print(f"Y_{args.K}: {Y}")


#计算密度信息fx
def compute_density_info(Y,word):
    distance = Y
    sorted_distances_for_token = sorted_cl100_emb.get(word, None) # 从 sorted_distance_data 中获取 origin_token 对应的距离信息列表
    distances_only = np.array([item[1] for item in sorted_distances_for_token]) # 提取 sorted_distances_for_token 列表中的距离部分，存储到 NumPy 数组
    index = np.searchsorted(distances_only, distance) # 找到 distance 在 distances_only 数组中的插入位置 index，保持数组的升序顺序
    f_x= int(index)

    return f_x


if not os.path.exists(f'data/eps_fx_{args.eps_fx}_K{args.K}/density_info.json'):
    word_list = list(token_to_vector_dict.keys())  # 包含所有词汇
    dict={}
    for word in word_list:
        dict[word] = compute_density_info(Y,word)

    with open(f'data/eps_fx_{args.eps_fx}_K{args.K}/density_info.json', 'w') as f:
        json.dump(dict, f)


density_info={}
with open(f"data/eps_fx_{args.eps_fx}_K{args.K}/density_info.json", 'r') as f:
    density_info = json.load(f)
    # print(density_info)

# 计算局部敏感度
def compute_local_sensitivity(Y,word):
    rho_x= density_info[word]
    # 从 sorted_cl100_emb 中获取 word 对应的距离信息列表
    sorted_distances_for_token = sorted_cl100_emb.get(word, None)
    # 获取离word最近的K个token
    closest_k_words = [item[0] for item in sorted_distances_for_token[:args.K] ]

    local_sensitivities = [
        abs(rho_x - density_info[token]) for token in closest_k_words
    ]

    # local_sensitivities = [abs(rho_x - density_info[key]) for key in token_to_vector_dict.keys()]
    local_sensitivity=max(local_sensitivities)
    return local_sensitivity


if not os.path.exists(f'data/eps_fx_{args.eps_fx}_K{args.K}/local_sensitivity.json'):
    word_list = list(token_to_vector_dict.keys())  # 包含所有词汇
    dict={}
    for word in word_list:
        dict[word] = compute_local_sensitivity(Y,word)

    with open(f'data/eps_fx_{args.eps_fx}_K{args.K}/local_sensitivity.json', 'w') as f:
        json.dump(dict, f)


local_dict={}
with open(f"data/eps_fx_{args.eps_fx}_K{args.K}/local_sensitivity.json", 'r') as f:
    local_dict = json.load(f)
    # print(local_dict)


# 计算平滑敏感度
def distance(x, y):
    a = token_to_vector_dict[x]
    b = token_to_vector_dict[y]
    return np.linalg.norm(a - b)

def compute_smooth_sensitivity(word, epsilon):

    sorted_distances_for_token = sorted_cl100_emb.get(word, None)
    # 获取离word最近的K个token
    closest_k_words = [item[0] for item in sorted_distances_for_token[:args.K] ]

    beta=epsilon/(2*math.log(2/delta))

    smooth_sensitivity = max(
        local_dict[token] * math.exp(-beta * distance(word, token))
        for token in closest_k_words
    )
    return smooth_sensitivity


if not os.path.exists(f'data/eps_fx_{args.eps_fx}_K{args.K}/smooth_sensitivity.json'):
    word_list = list(token_to_vector_dict.keys())  # 包含所有词汇
    dict={}
    for word in word_list:
        dict[word] = compute_smooth_sensitivity(word,args.eps_fx)

    with open(f'data/eps_fx_{args.eps_fx}_K{args.K}/smooth_sensitivity.json', 'w') as f:
        json.dump(dict, f)


smooth_dict={}
with open(f"data/eps_fx_{args.eps_fx}_K{args.K}/smooth_sensitivity.json", 'r') as f:
    smooth_dict = json.load(f)
    # print(smooth_dict)


# 根据平滑敏感度添加lap噪声
# def add_laplace_noise(Y,word, epsilon):

#     beta_value =smooth_dict[word] / epsilon
#     noise = np.random.laplace(0, beta_value)

#     return density_info[word] + noise

def calculate_sigma(sensitivity, epsilon, delta):
    """
    根据灵敏度、隐私预算和置信度参数计算高斯噪声的标准差。
    """
    return (sensitivity * np.sqrt(2 * np.log(1.25 / delta))) / epsilon



def add_gaussian_noise(Y, word, epsilon):
    # 根据 epsilon 计算标准差
    sensitivity = smooth_dict[word]  # 假设 smooth_dict 中存的是灵敏度
    sigma_value = calculate_sigma(sensitivity, epsilon, delta)
    # 添加高斯噪声
    noise = np.random.normal(0, sigma_value)
    return density_info[word] + noise


if not os.path.exists(f'data/eps_fx_{args.eps_fx}_K{args.K}/F_x.json'):
    word_list = list(token_to_vector_dict.keys())  # 包含所有词汇

    dict={}
    for word in word_list:
        dict[word] = add_gaussian_noise(Y,word,args.eps_fx)
        # dict[word] = add_laplace_noise(Y,word,args.eps_fx)

    clipped_F_x=dict
    values = list(clipped_F_x.values())
    # 计算最小值和最大值
    min_value = min(values)
    max_value = max(values)

    # 使用 Min-Max 归一化公式计算归一化后的值
    normalized_dict = {
        k: (v - min_value) / (max_value - min_value)
        for k, v in clipped_F_x.items()
    }

    with open(f'data/eps_fx_{args.eps_fx}_K{args.K}/F_x.json', 'w') as f:
        json.dump(dict, f)

    with open(f'data/eps_fx_{args.eps_fx}_K{args.K}/clipped_F_x.json', 'w') as f:
        json.dump(clipped_F_x, f)

    with open(f'data/eps_fx_{args.eps_fx}_K{args.K}/normalized_F_x.json', 'w') as f:
        json.dump(normalized_dict, f)


F_x={}
with open(f"data/eps_fx_{args.eps_fx}_K{args.K}/F_x.json", 'r') as f:
    F_x = json.load(f)
    # print(F_x)

print(f'F_x: {F_x["A"]}')

clipped_F_x={}
with open(f"data/eps_fx_{args.eps_fx}_K{args.K}/clipped_F_x.json", 'r') as f:
    clipped_F_x = json.load(f)
    # print(clipped_F_x)

print(f'clipped_F_x: {clipped_F_x["A"]}')


neighbor_values = list(clipped_F_x.values())
# 统计每个邻接词数量出现的次数
count_of_neighbors = Counter(neighbor_values)

x = list(count_of_neighbors.keys())
y = list(count_of_neighbors.values())

# 绘制直方图
plt.bar(x, y)
plt.xlabel('Number of Neighbors')
plt.ylabel('Number of Tokens')
plt.title(f'Histogram of Neighbor Counts per Token (eps_fx={args.eps_fx}, K={args.K})')
plt.ylim(0,20)

plt.grid()

# 保存图像或者显示
plt.savefig(f'fig/eps_fx_{args.eps_fx}_K{args.K}_Fx.png')  # 保存图像



normalized_F_x={}
with open(f"data/eps_fx_{args.eps_fx}_K{args.K}/normalized_F_x.json", 'r') as f:
    normalized_F_x = json.load(f)
    # print(normalized_F_x)

# 获取字典中的值


print(f'normalized_F_x: {normalized_F_x["A"]}')









