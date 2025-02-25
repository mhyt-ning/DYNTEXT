#!/bin/bash         
# conda activate InferDPT
export CUDA_VISIBLE_DEVICES=6


# python3 -u ./0.Y.py > log/0.Y.log 2>&1 &



#得到加噪密度信息
# eps_fx_value=0.5
# k_value=100
# python3 -u ./0.F_x.py  --eps_fx $eps_fx_value --K $k_value > log/eps_fx_${eps_fx_value}_K${k_value}.log 2>&1 &


#扰动
eps_fx_value=0.5
k_value=100
eps_values=(2.5)  # 替换为您想要的 eps_value 列表

for eps_value in "${eps_values[@]}"
do
    # 运行 Python 脚本并将日志输出到对应的文件中
    python3 -u ./1.perturbation.py --eps $eps_value --eps_fx $eps_fx_value --K $k_value > log/pubmedqa/perturbation_epsfx${eps_fx_value}_eps${eps_value}_K${k_value}.log 2>&1 &
done


#提取
# eps_fx_value=0.5
# k_value=20
# eps_values=(1.5 2.5)
# python3 -u ./3.extraction.py  > log/pubmedqa/extraction_llm.log 2>&1 &

