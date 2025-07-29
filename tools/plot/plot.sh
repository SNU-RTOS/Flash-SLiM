#!/bin/bash

# # TTFT / ATNT / AMPF 다 보여줌
# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/Thread1_8_128_512in_32out \
#     --ram-sizes 256M 512M 1G 2G 4G 8G

# 256M 제외외 - 이게 진짜임!!! 25-02-23
python3.10 show_plot.py \
    --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/result_0325 \
    --ram-sizes 512M 1G 2G 4G 8G \
    --input-tokens 8 128 512

# ### RUBIKPi T1C4 T4C4
# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/RUBIKPi_T1C1/TC2G \
#     --ram-sizes T1C1 T1C2 T1C4 T2C1 T2C2 T2C4 T4C1 T4C2 T4C4

# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/RUBIKPi_T1C1/TC2G \
#     --ram-sizes T4C1 T4C2 T4C4

# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/RUBIKPi_T1C1/TC2G \
#     --ram-sizes T1C1 T2C1 T4C1

# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/RUBIKPi_T1C1/TC2G \
#     --ram-sizes T1C1 T4C4
# ###

# ### Jenkins 실험 결과
# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/log_backup_perf \
#     --ram-sizes 1G 2G 4G 8G
# ###
# python3.10 show_plot_old.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/log_backup_perf \
#     --ram-sizes 1G 2G 4G 8G



# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app \
#     --ram-sizes 8G_No_Preload 8G_Preload


# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/Naive_4Thread_4Core \
#     --ram-sizes 2G Reordering

# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/RUBIKPi_8G \
#     --ram-sizes T1C4_Naive T1C4_Preload T1C4_NoMmap

# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/RUBIKPi_8G \
#     --ram-sizes T1C1_Naive T1C1_Preload T1C1_NoMmap

# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/Gemma_Naive_4Thread_4Core \
#     --ram-sizes 512M 1G 2G 4G 8G

# # 여기서부턴 가짜임! 25-02-23
# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/New_Naive \
#     --ram-sizes 512M 1G 2G 4G 8G

# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/Direct_IO \
#     --ram-sizes 512M 1G 2G 4G 8G

# python3.10 show_plot.py \
#     --base-dir /home/rtos/workspace/ghpark/LLM-Inference-app/evaluation/Thread1_output \
#     --ram-sizes Naive Reordering Pinning Reorder+Pin