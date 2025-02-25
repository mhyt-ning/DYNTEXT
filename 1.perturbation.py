from func import *
import openai
import pandas as pd
import math
from args import *



parser = get_parser()
args = parser.parse_args()


with open("../data/cl100_embeddings.json", 'r') as f:
    cl100_emb = json.load(f)
    vector_data_json = {k: cl100_emb[k] for k in list(cl100_emb.keys())[:11000]} #词库总共有11000个词
    cl100_emb = None
    # 将这些键值对中的向量数据转换为 NumPy 数组，以方便进行数学和向量操作
    token_to_vector_dict = {token: np.array(vector) for token, vector in vector_data_json.items()}

# print(token_to_vector_dict)

if not os.path.exists(f'../data/sorted_cl100_embeddings.json'):
    init_func(1.0, token_to_vector_dict)
with open('../data/sorted_cl100_embeddings.json', 'r') as f1:
    sorted_cl100_emb = json.load(f1)
with open('../data/sensitivity_of_embeddings.json', 'r') as f:
    sen_emb = np.array(json.load(f)) 
with open(f"data/eps_fx_{args.eps_fx}_K{args.K}/normalized_F_x.json", 'r') as f2:
    normalized_F_x = json.load(f2)



data=pd.read_csv(args.raw_path)
print(f'epsilon_fx is {args.eps_fx}; epsilon_perturbed is {args.eps}.')

print(f'length: {len(data)}')

replace_dict=[]

for i in range(len(data)):
    print(f'第{i}条数据：————————————————————————————————————————————————')
    raw_document =str(data.loc[i,args.text])
    print(f"raw_document: {raw_document}")
    raw_tokens = get_first_50_tokens(raw_document)
    print(f"raw_tokens: {raw_tokens}")
    perturbed_tokens = perturb_sentence(raw_tokens, args.eps, args.model, token_to_vector_dict, sorted_cl100_emb,
                                        sen_emb,normalized_F_x,replace_dict)
    print(f'perturbed_tokens: {perturbed_tokens}')
    data.loc[i, args.text] = perturbed_tokens
    print(f'扰动结束。————————————————————————————————————————————')


with open(f'output/{args.task}/replace_dict_myway{args.eps}.json', 'w') as f:
    json.dump(replace_dict, f, indent=4)

print('处理完毕。')
perturbed_path=f'output/{args.task}/perturbed_tokens_myway{args.eps_fx}+{args.eps}_K{args.K}.csv'

data.to_csv(perturbed_path,index=False)