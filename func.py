import json
import string
import tiktoken
import random
from sklearn.metrics.pairwise import euclidean_distances
import tqdm
from decimal import getcontext
import numpy as np
import json
from transformers import GPT2Tokenizer
import os 
import pandas as pd
from args import *



parser = get_parser()
args = parser.parse_args()

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# 获取英语停用词列表
stop_words = set(stopwords.words('english'))

def is_stop_word(word):
    return word.lower() in stop_words

getcontext().prec = 100
#os.environ["http_proxy"] = "http://127.0.0.1:10809"
#os.environ["https_proxy"] = "http://127.0.0.1:10809"

tokenizer = GPT2Tokenizer.from_pretrained("../gpt2",force_download=True)

def get_first_50_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained("../gpt2")
    tokens = tokenizer.tokenize(text)
    first_50_tokens = tokens[:50]
    tokenized_string = tokenizer.convert_tokens_to_string(first_50_tokens)
    return tokenized_string

def get_50_150_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained("../gpt2")
    tokens = tokenizer.tokenize(text)
    after_50_tokens = tokens[50:150]
    tokenized_string = tokenizer.convert_tokens_to_string(after_50_tokens)
    return tokenized_string

def get_first_500_tokens(text):
    tokens = tokenizer.tokenize(text)
    first_500_tokens = tokens[:500]
    tokenized_string = tokenizer.convert_tokens_to_string(first_500_tokens)
    return tokenized_string

def get_first_100_tokens(text):
    tokenizer = GPT2Tokenizer.from_pretrained("../gpt2")
    tokens = tokenizer.tokenize(text)
    first_100_tokens = tokens[:100]
    tokenized_string = tokenizer.convert_tokens_to_string(first_100_tokens)
    return tokenized_string

def calculate_distance(i, j, vector_matrix, pb):
    distance = euclidean_distances(vector_matrix[i].reshape(1, -1).astype(np.longdouble), 
                                   vector_matrix[j].reshape(1, -1).astype(np.longdouble))
    pb.update(1)
    return i, j, distance[0, 0]

punctuation_string = string.punctuation
punctuation_list = list(punctuation_string)

def generate_tasks(n_vectors):
    for i in range(n_vectors):
        for j in range(i + 1, n_vectors):
            yield (i, j)


# 添加lap噪声
def add_laplace_noise_to_vector(vector, epsilon, delta_f_new=None):
    vector = np.asarray(vector, dtype=np.longdouble) # 将输入向量转换为高精度数据类型
    # print(f'vector:{vector}')
    if not os.path.exists(f'../data/sorted_cl100_embeddings.json'): # 检查预计算的排序距离数据是否存在，若不存在
        # 键为词汇，值为相应的嵌入向量。
        with open("../data/cl100_embeddings.json", 'r') as f:
            data_t=json.load(f)

            random.seed(42)
            random_keys = random.sample(list(data_t.keys()), 11000)  # 随机取100个词


            data = {k: data_t[k] for k in random_keys}
            data_t=None
        word_list = list(data.keys()) # 包含所有词汇
        vector_matrix = np.array(list(data.values())) # 是嵌入向量的矩阵表示，每一行是一个词的向量
        data=None
        n_vectors = len(word_list)
        distance_matrix = np.zeros((n_vectors, n_vectors)) # 初始化距离矩阵 distance_matrix，用于存储每对词汇之间的距离
        total_tasks = (n_vectors * (n_vectors - 1)) // 2
        results = [None] * total_tasks
        if not os.path.exists(f'../data/temp_distance_json_path.json'):
            print('no')
            with tqdm.tqdm(total=int(n_vectors * (n_vectors - 1) / 2)) as pb:
                pb.set_description('Inference process')
                # 使用 generate_tasks 函数生成任务列表，每个任务计算词汇对之间的距离。，并存储在 results 列表中。
                print(n_vectors)
                tasks = list(generate_tasks(n_vectors))
                for index, task in enumerate(tasks):
                    try:
                        results[index] = calculate_distance(task[0], task[1], vector_matrix, pb)
                    except Exception as e:
                        print(f"Task at index {index} failed with exception {e}")
                # 更新距离矩阵
                for i, j, distance in results:
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
            # 保存距离矩阵和距离数据字典
            temp_distance_matrix =distance_matrix
            temp_distance_dict_matrix = {}
            for i, word1 in enumerate(word_list):
                for j, word2 in enumerate(word_list):
                    pair = tuple(sorted([word1, word2]))
                    if pair in temp_distance_dict_matrix:
                        continue
                    temp_distance_dict_matrix[str(pair)] = float(temp_distance_matrix[i, j])

            print(temp_distance_dict_matrix)
            with open('../data/temp_distance_json_path.json', 'w') as f:
                json.dump(temp_distance_dict_matrix, f)
        if os.path.exists(f'../data/temp_distance_json_path.json'):
            with open('../data/temp_distance_json_path.json', 'r') as f:
                temp_distance_dict_matrix = json.load(f)
            word_to_index = {}
            with tqdm.tqdm(total=len(word_list)) as pbwi:
                pbwi.set_description('word_to_index process')
                for idx, word in enumerate(word_list):
                    word_to_index[word] = idx
                    pbwi.update(1)
            n = len(word_list)
            temp_distance_matrix = np.zeros((n, n))
            with tqdm.tqdm(total=len(temp_distance_dict_matrix)) as pbm:
                pbm.set_description('')
                for key, value in temp_distance_dict_matrix.items():
                    word1, word2 = tuple(key.strip("()").split(", "))
                    i = word_to_index[word1.strip("'")]
                    j = word_to_index[word2.strip("'")]
                    temp_distance_matrix[i, j] = value
                    temp_distance_matrix[j, i] = value
                    pbm.update(1)
            # 构建排序的距离字典。按照每个词的距离对其他词进行排序，构建 sorted_distance_dict_matrix，用于快速查找最近的词汇
            sorted_distance_dict_matrix = {}
            with tqdm.tqdm(total=n) as pbm:
                pbm.set_description('Sorted process')
                for i, word in enumerate(word_list):
                    sorted_indices = np.argsort(temp_distance_matrix[i])
                    sorted_words = [(word_list[j], temp_distance_matrix[i, j]) for j in sorted_indices]
                    sorted_distance_dict_matrix[word] = sorted_words
                    pbm.update(1)

        with open('../data/sorted_cl100_embeddings.json', 'w') as f:
            json.dump(sorted_distance_dict_matrix, f)

    # 计算或加载向量的敏感度 (delta_f_new): ！！！！！！！！
    if not os.path.exists(f'../data/sensitivity_of_embeddings.json'): # 不存在，计算每个维度上的最大差值来确定敏感度。
        json_path = "../data/cl100_embeddings.json"
        with open(json_path, 'r') as f:
            cl100_emb = json.load(f)
            random.seed(42)
            random_keys = random.sample(list(cl100_emb.keys()), 11000)  # 随机取100个词
            vector_data_json = {k: cl100_emb[k] for k in random_keys}
            cl100_emb = None

            # vector_data_json = json.load(f)
        word_list = list(vector_data_json.keys())
        vector_matrix = np.array(list(vector_data_json.values()))
        print(f'vector_matrix: {vector_matrix}')
        n_dimensions = vector_matrix.shape[1]
        print(f'n_dimensions: {n_dimensions}')
        delta_f_new = np.zeros(n_dimensions)
        print(f'delta_f_new: {delta_f_new}')
        for dim in tqdm.trange(n_dimensions):
            dim_data = vector_matrix[:, dim]
            sorted_dim_data = np.sort(dim_data)
            differences =sorted_dim_data[-1]-sorted_dim_data[0]
            delta_f_new[dim] = differences
        delta_f_new_json_path = '../data/sensitivity_of_embeddings.json'
        with open(delta_f_new_json_path, 'w') as f:
            json.dump(delta_f_new.tolist(), f)
    else:
        if delta_f_new is None:
            with open('../data/sensitivity_of_embeddings.json', 'r') as f:
                delta_f_new = np.array(json.load(f))

    # 计算拉普拉斯噪声的尺度参数 beta_values。如果 epsilon 小于 2，直接使用；否则，使用计算出的 tt 值。
    tt=0
    if (epsilon*19.064721649556482-38.1294334077209)>0:
        tt=0.01658160142016071*np.log(epsilon*19.064721649556482-38.1294334077209)+9.311083811697406
    if epsilon <2:
        print(f'delta_f_new: {delta_f_new}')
        beta_values = delta_f_new/epsilon
    else:
        beta_values = delta_f_new/tt
    beta_values = beta_values.astype(np.longdouble)
    # 遍历向量的每个维度，为其添加从拉普拉斯分布中生成的噪声。噪声的强度取决于 beta_values，从而控制添加噪声的程度。
    noisy_vector = np.zeros_like(vector, dtype=np.longdouble)
    for dim in range(len(vector)):
        noise = np.random.laplace(0, beta_values[dim])
        noisy_vector[dim] = vector[dim] + noise
    return noisy_vector.astype(float) 


def perturb_sentence(sent, epsilon, model, token_to_vector_dict,sorted_distance_data,delta_f_new,normalized_F_x,replace_dict):
    # 将句子进行 tokenization（分词）和编码
    print("NEW")
    print(f'orginal_epsilon:{epsilon}')

    tokens = sent.split()

    new_tokens=[]
    Delta_u = 1.0  
    exp_factor = epsilon / (2 * Delta_u)
    for origin_token in tokens:
        replace_dict.append(origin_token)
        print(f"Original token: {origin_token}")
        if(origin_token.isnumeric()):
            a=str(random.randint(1, 1000))
            new_tokens.append(a)
            replace_dict.append(a)
            continue
        if(origin_token[0]==' '):
            origin_token=origin_token[1:]  

        ###################################
        #保留停用词
        if is_stop_word(origin_token):
            new_tokens.append(origin_token)
            print(f"New token: {origin_token}")
            replace_dict.append(origin_token)
            continue  
        ###################################
  
        # 从 token_to_vector_dict 字典中获取 origin_token 对应的向量嵌入 origin_embed
        origin_embed = token_to_vector_dict.get(origin_token, None)
        if origin_embed is None:
            new_tokens.append(origin_token)
            print("不在词库中")
            print(f"New token: {origin_token}")
            replace_dict.append(origin_token)
            continue
        
        #利用密度信息放缩自适应调整K
        midu=normalized_F_x.get(origin_token, None)


        index=int(0.5*args.K-(midu-0.5)*args.K)+1


        sorted_distances_for_token = sorted_distance_data.get(origin_token, None)
        if sorted_distances_for_token is None:
            raise ValueError("No distance data available for the given token.")
        distances_only = np.array([item[1] for item in sorted_distances_for_token])

        # 根据索引 index 反推出距离
        if 0 <= index < len(distances_only):  # 确保索引有效
            distance = distances_only[index]


        # 获取所有距离比 distance 小的 token及其距离
        print(f"index: {index}")
        close_tokens = [item[0] for item in sorted_distances_for_token[:index] ]
        close_distances = np.array([item[1] for item in sorted_distances_for_token[:index]])
        if not close_tokens:
            replace_dict.append(' ')
            continue
        # 计算未归一化的概率，基于 distance 和 close_distances 的差距以及一个扩展因子 exp_factor。
        unnormalized_probabilities = np.exp(exp_factor * ((distance-close_distances)/distance))
        total_unnormalized_prob = np.sum(unnormalized_probabilities) # 计算所有未归一化概率的总和
        probabilities = unnormalized_probabilities / total_unnormalized_prob # 将未归一化的概率归一化
        selected_token = np.random.choice(close_tokens, p=probabilities) # 根据 probabilities 分布，随机选择一个接近的 token
        new_tokens.append(selected_token)
        print(f"New token: {selected_token}")
        replace_dict.append(selected_token)
    sanitized_sent = ' '.join(new_tokens)
    return sanitized_sent

def init_func(epsilon,token_to_vector_dict):
    origin_embed = token_to_vector_dict.get('input', None)
    add_laplace_noise_to_vector(origin_embed,epsilon)
    print("aaaaaaaaaaaaaaa")
