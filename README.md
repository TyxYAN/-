![image](https://github.com/TyxYAN/-/assets/101959829/f9d52afc-f4d3-4e93-b60d-612c850aefc6)![image](https://github.com/TyxYAN/-/assets/101959829/f9d52afc-f4d3-4e93-b60d-612c850aefc6)# 书生·浦语大模型全链路开源体系
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


# "茴香豆":零代码搭建你的 RAG 智能助理
![image](https://github.com/TyxYAN/-/assets/101959829/d7740143-3b9e-45ab-8e63-e01746f19448)
RAG（Retrieval Augmented Generation）技术，通过检索与用户输入相关的信息片段，并结合外部知识库来生成更准确、更丰富的回答。解决 LLMs 在处理知识密集型任务时可能遇到的挑战, 如幻觉、知识过时和缺乏透明、可追溯的推理过程等。提供更准确的回答、降低推理成本、实现外部记忆。
RAG 能够让基础模型实现非参数知识更新，无需训练就可以掌握新领域的知识。本次课程选用的茴香豆应用，就应用了 RAG 技术，可以快速、高效的搭建自己的知识领域助手。

## 环境配置
studio-conda -o internlm-base -t InternLM2_Huixiangdou
![image](https://github.com/TyxYAN/-/assets/101959829/bb101f3c-fd02-4d6f-9460-82532dfb99b6)

conda env list
![image](https://github.com/TyxYAN/-/assets/101959829/adcd99bf-3f38-4fa2-99db-9c827a2d9bf9)

## 下载基础文件

创建模型文件夹
cd /root && mkdir models

ln -s /root/share/new_models/maidalun1020/bce-embedding-base_v1 /root/models/bce-embedding-base_v1
ln -s /root/share/new_models/maidalun1020/bce-reranker-base_v1 /root/models/bce-reranker-base_v1

ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/models/internlm2-chat-7b

##  下载安装茴香豆
pip install protobuf==4.25.3 accelerate==0.28.0 aiohttp==3.9.3 auto-gptq==0.7.1 bcembedding==0.1.3 beautifulsoup4==4.8.2 einops==0.7.0 faiss-gpu==1.7.2 langchain==0.1.14 loguru==0.7.2 lxml_html_clean==0.1.0 openai==1.16.1 openpyxl==3.1.2 pandas==2.2.1 pydantic==2.6.4 pymupdf==1.24.1 python-docx==1.1.0 pytoml==0.1.21 readability-lxml==0.8.1 redis==5.0.3 requests==2.31.0 scikit-learn==1.4.1.post1 sentence_transformers==2.2.2 textract==1.6.5 tiktoken==0.6.0 transformers==4.39.3 transformers_stream_generator==0.0.5 unstructured==0.11.2

cd /root
git clone https://github.com/internlm/huixiangdou && cd huixiangdou
git checkout ded0551
![image](https://github.com/TyxYAN/-/assets/101959829/b10d06c2-e841-45ac-9f1d-6def42a73459)
## 使用茴香豆搭建 RAG 助手
修改配置文件
用已下载模型的路径替换 /root/huixiangdou/config.ini 文件中的默认模型，需要修改 3 处模型地址，分别是:

命令行输入下面的命令，修改用于向量数据库和词嵌入的模型

sed -i '6s#.*#embedding_model_path = "/root/models/bce-embedding-base_v1"#' /root/huixiangdou/config.ini
用于检索的重排序模型

sed -i '7s#.*#reranker_model_path = "/root/models/bce-reranker-base_v1"#' /root/huixiangdou/config.ini
和本次选用的大模型

sed -i '29s#.*#local_llm_path = "/root/models/internlm2-chat-7b"#' /root/huixiangdou/config.ini
修改好的配置文件如下图所示：

## 创建知识库
使用 InternLM 的 Huixiangdou 文档作为新增知识数据检索来源，在不重新训练的情况下，打造一个 Huixiangdou 技术问答助手。

首先，下载 Huixiangdou 语料：

cd /root/huixiangdou && mkdir repodir
git clone https://github.com/internlm/huixiangdou --depth=1 repodir/huixiangdou

提取知识库特征，创建向量数据库。数据库向量化的过程应用到了 LangChain 的相关模块，默认嵌入和重排序模型调用的网易 BCE 双语模型，如果没有在 config.ini 文件中指定本地模型路径，茴香豆将自动从 HuggingFace 拉取默认模型。

除了语料知识的向量数据库，茴香豆建立接受和拒答两个向量数据库，用来在检索的过程中更加精确的判断提问的相关性，这两个数据库的来源分别是：

接受问题列表，希望茴香豆助手回答的示例问题
存储在 huixiangdou/resource/good_questions.json 中
拒绝问题列表，希望茴香豆助手拒答的示例问题
存储在 huixiangdou/resource/bad_questions.json 中
运行下面的命令，增加茴香豆相关的问题到接受问题示例中：

cd /root/huixiangdou
mv resource/good_questions.json resource/good_questions_bk.json

echo '[
    "mmpose中怎么调用mmyolo接口",
    "mmpose实现姿态估计后怎么实现行为识别",
    "mmpose执行提取关键点命令不是分为两步吗，一步是目标检测，另一步是关键点提取，我现在目标检测这部分的代码是demo/topdown_demo_with_mmdet.py demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth   现在我想把这个mmdet的checkpoints换位yolo的，那么应该怎么操作",
    "在mmdetection中，如何同时加载两个数据集，两个dataloader",
    "如何将mmdetection2.28.2的retinanet配置文件改为单尺度的呢？",
    "1.MMPose_Tutorial.ipynb、inferencer_demo.py、image_demo.py、bottomup_demo.py、body3d_pose_lifter_demo.py这几个文件和topdown_demo_with_mmdet.py的区别是什么，\n2.我如果要使用mmdet是不是就只能使用topdown_demo_with_mmdet.py文件，",
    "mmpose 测试 map 一直是 0 怎么办？",
    "如何使用mmpose检测人体关键点？",
    "我使用的数据集是labelme标注的，我想知道mmpose的数据集都是什么样式的，全都是单目标的数据集标注，还是里边也有多目标然后进行标注",
    "如何生成openmmpose的c++推理脚本",
    "mmpose",
    "mmpose的目标检测阶段调用的模型，一定要是demo文件夹下的文件吗，有没有其他路径下的文件",
    "mmpose可以实现行为识别吗，如果要实现的话应该怎么做",
    "我在mmyolo的v0.6.0 (15/8/2023)更新日志里看到了他新增了支持基于 MMPose 的 YOLOX-Pose，我现在是不是只需要在mmpose/project/yolox-Pose内做出一些设置就可以，换掉demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py 改用mmyolo来进行目标检测了",
    "mac m1从源码安装的mmpose是x86_64的",
    "想请教一下mmpose有没有提供可以读取外接摄像头，做3d姿态并达到实时的项目呀？",
    "huixiangdou 是什么？",
    "使用科研仪器需要注意什么？",
    "huixiangdou 是什么？",
    "茴香豆 是什么？",
    "茴香豆 能部署到微信吗？",
    "茴香豆 怎么应用到飞书",
    "茴香豆 能部署到微信群吗？",
    "茴香豆 怎么应用到飞书群",
    "huixiangdou 能部署到微信吗？",
    "huixiangdou 怎么应用到飞书",
    "huixiangdou 能部署到微信群吗？",
    "huixiangdou 怎么应用到飞书群",
    "huixiangdou",
    "茴香豆",
    "茴香豆 有哪些应用场景",
    "huixiangdou 有什么用",
    "huixiangdou 的优势有哪些？",
    "茴香豆 已经应用的场景",
    "huixiangdou 已经应用的场景",
    "huixiangdou 怎么安装",
    "茴香豆 怎么安装",
    "茴香豆 最新版本是什么",
    "茴香豆 支持哪些大模型",
    "茴香豆 支持哪些通讯软件",
    "config.ini 文件怎么配置",
    "remote_llm_model 可以填哪些模型?"
]' > /root/huixiangdou/resource/good_questions.json
再创建一个测试用的问询列表，用来测试拒答流程是否起效：

cd /root/huixiangdou

echo '[
"huixiangdou 是什么？",
"你好，介绍下自己"
]' > ./test_queries.json
在确定好语料来源后，运行下面的命令，创建 RAG 检索过程中使用的向量数据库：

完成后，Huixiangdou 相关的新增知识就以向量数据库的形式存储在 workdir 文件夹下。
检索过程中，茴香豆会将输入问题与两个列表中的问题在向量空间进行相似性比较，判断该问题是否应该回答，避免群聊过程中的问答泛滥。确定的回答的问题会利用基础模型提取关键词，在知识库中检索 top K 相似的 chunk，综合问题和检索到的 chunk 生成答案

## 运行茴香豆知识助手
现在，让我们来测试一下效果：

命令行运行：

# 填入问题
sed -i '74s/.*/    queries = ["huixiangdou 是什么？", "茴香豆怎么部署到微信群", "今天天气怎么样？"]/' /root/huixiangdou/huixiangdou/main.py

# 运行茴香豆
cd /root/huixiangdou/
python3 -m huixiangdou.main --standalone
![image](https://github.com/TyxYAN/-/assets/101959829/4358d8c6-a88b-4c5c-860a-edf714e1ec96)

