# 书生·浦语大模型全链路开源体系
书生·浦语大模型的发展历程
![5739af4a63d7152820e3c44091626378](https://github.com/TyxYAN/-/assets/101959829/5b4db4ec-38bc-4c81-a4e5-bc1bdb464e80)
# 轻松分钟玩转书生·浦语大模型趣味 Demo
使用 InternLM2-Chat-1.8B 模型生成 300 字的小故事
创建开发机
![image](https://github.com/TyxYAN/-/assets/101959829/685cb172-7e31-4966-91e0-5d7b8c7386e6)


## 环境配置
进入开发机后，在 terminal 中输入环境配置命令 

studio-conda -o internlm-base -t demo

![image](https://github.com/TyxYAN/-/assets/101959829/5cbf68d6-dd3b-47df-b399-e2ca97c94ea2)

配置完成后，进入到新创建的 conda 环境之中：
conda activate demo

输入以下命令，完成环境包的安装：
Python
pip install huggingface-hub==0.17.3
pip install transformers==4.34 
pip install psutil==5.9.8
pip install accelerate==0.24.1
pip install streamlit==1.32.2 
pip install matplotlib==3.8.3 
pip install modelscope==1.9.5
pip install sentencepiece==0.1.99
## 下载 InternLM2-Chat-1.8B 模型
按路径创建文件夹，并进入到对应文件目录中：

mkdir -p /root/demo
touch /root/demo/cli_demo.py
touch /root/demo/download_mini.py
cd /root/demo


打开 /root/demo/download_mini.py 文件，copy以下代码：
import os
from modelscope.hub.snapshot_download import snapshot_download

 创建保存模型目录
os.system("mkdir /root/models")

 save_dir是模型保存到本地的目录
save_dir="/root/models"

snapshot_download("Shanghai_AI_Laboratory/internlm2-chat-1_8b", 
                  cache_dir=save_dir, 
                  revision='v1.1.0')

执行命令，下载模型参数文件：
python /root/demo/download_mini.py

![image](https://github.com/TyxYAN/-/assets/101959829/d33437ca-1110-49f6-b00d-706ed1667f26)


## 运行 cli_demo
双击打开 /root/demo/cli_demo.py 文件，复制以下代码：

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name_or_path = "/root/models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, device_map='cuda:0')
model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='cuda:0')
model = model.eval()

system_prompt = """You are an AI assistant whose name is InternLM (书生·浦语).
- InternLM (书生·浦语) is a conversational language model that is developed by Shanghai AI Laboratory (上海人工智能实验室). It is designed to be helpful, honest, and harmless.
- InternLM (书生·浦语) can understand and communicate fluently in the language chosen by the user such as English and 中文.
"""

messages = [(system_prompt, '')]

print("=============Welcome to InternLM chatbot, type 'exit' to exit.=============")

while True:
    input_text = input("\nUser  >>> ")
    input_text = input_text.replace(' ', '')
    if input_text == "exit":
        break

    length = 0
    for response, _ in model.stream_chat(tokenizer, input_text, messages):
        if response is not None:
            print(response[length:], flush=True, end="")
            length = len(response)

输入命令，执行 Demo 程序：
conda activate demo
python /root/demo/cli_demo.py

# 等待模型加载完成，键入内容示例：
## 请创作一个 300 字的小故事
得到的输出：
![image](https://github.com/TyxYAN/-/assets/101959829/14b92606-20ab-4a62-a8f9-51dd754ac847)

## 使用书生·浦语 Web 和浦语对话
使用书生·浦语 Web 和浦语对话，和书生·浦语对话，并找到书生·浦语 1 处表现不佳的案例(比如指令遵循表现不佳的案例)
![image](https://github.com/TyxYAN/-/assets/101959829/97e4bb51-2b47-47bb-87cb-cb6f2102f2c7)





