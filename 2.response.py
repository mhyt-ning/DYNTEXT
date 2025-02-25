import openai
import pandas as pd
from func import *
import time
import os

def text_generaton_with_black_box_LLMs(prompt,tem):
    res=openai.ChatCompletion.create(model='gpt-4',
                            messages=[
                            {'role': 'user', 'content': prompt}],
                            max_tokens=150,
                            temperature=tem,
                                    )
    return res

# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"


# train_path='output/cnn_daily/perturbed_tokens_baseline_1000.csv'
train_path='./output/cnn_daily/perturbed_tokens_myway_1000.csv'
data=pd.read_csv(train_path)

result=data

max_retries = 10
retry_delay = 5  # seconds



for i in range(0,len(data)):
    print(i)
    perturbed_tokens = data.loc[i, 'text']
    prompt = """Your task is to extend Prefix Text.
        - Prefix Text:""" + perturbed_tokens + """
        \n\n Provide only your Continuation.
        - Continuation:"""

    for j in range(max_retries):
        try:
            response = text_generaton_with_black_box_LLMs(prompt,0.5)
            response_text = get_first_100_tokens(response['choices'][0]['message']['content'])
            result.loc[i, 'text'] = response_text
            print(response_text)
            break  # 如果调用成功，跳出循环
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(retry_delay)

result.to_csv('./output/cnn_daily/response_myway_1000.csv',index=False)
