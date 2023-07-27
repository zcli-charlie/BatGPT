# BatGPT
Bidirectional Autoregressive Talker from Generative Pre-trained Transformer

## 开源计划

### 模型

- [**BatGPT-15B-sirius**](https://huggingface.co/MLP-lab/BatGPT-15B-sirius): BatGPT第一个开源模型，具有 150 亿参数，在高质量中英文语料上进行双向自回归预训练得到，并进行了指令微调与强化对齐的学习，具有指令遵循能力、多轮对话能力、推理等能力。

- **mBatGPT-15B-sirius**: 具有图像，语音多模态理解能力的大模型，基于 BatGPT-15B-sirius 在500万图像文本，语音文本对上进行二阶预训练得到。


***

## Demo

我们提供了一个基于Streamlit实现的网页Demo，您可以使用streamlit运行本仓库中的batgpt_web_demo.py来打开网页Demo：

```bash
streamlit run batgpt_web_demo.py
```

***

## **BatGPT-15B-sirius**

### 介绍 (Introduction)

BatGPT-15B-sirius 是上海交通大学与武汉大学<font size=1>（或武汉大学与上海交通大学，排名不分先后）</font>联合自然语言处理团队设计、预训练、对齐的系列大型语言模型 [BatGPT](https://github.com/zcli-charlie/BatGPT) 中的一个开源可商用版本。
BatGPT系列模型中还包括BatGPT-30B-orion，BatGPT-70B-alhena，以及BatGPT-140B-menkalinan。

BatGPT-15B-sirius 包含 150 亿参数，在中英文 1T 语料上进行了预训练，在权威的中文和英文 benchmark 上均取得同不错的效果。BatGPT-15B-sirius 有如下几个特点：

  1. **支持长达32K的上下文**：BatGPT-15B-sirius 采用旋转位置编码RoPE，在预训练阶段采用 2048 序列长度，并且在指令微调阶段逐步扩展到了 32K 上下文。
  2. **高效的预训练目标与模型架构**：BatGPT-15B-sirius 采用双向自回归预训练目标，以提高对于训练数据的运用程度，并且基于 [Multi-Query Attention](http://arxiv.org/abs/1911.02150) 技术，在保证参数规模的前提下尽可能的减少推理显存的占用，提高推理速度。
  3. **商业友好的开放协议**：BatGPT-15B-sirius 的源码以及权重不仅支持自由的学术研究使用，也允许免费开源商用，助推大模型进一步帮助人类的日常生活。

BatGPT-15B-sirius is an open-source commercially available version of the series of large-scale language models [BatGPT](https://github.com/zcli-charlie/BatGPT), designed, pretrained, and aligned by the joint natural language processing teams of Shanghai Jiao Tong University and Wuhan University <font size=1>(or Wuhan University and Shanghai Jiao Tong University, in no particular order)</font>.

The BatGPT series of models also include BatGPT-30B-orion, BatGPT-70B-alhena, and BatGPT-140B-menkalinan.

BatGPT-15B-sirius contains 15 billion parameters and has been pretrained on 1T Chinese and English corpora. It achieves excellent performance on authoritative Chinese and English benchmarks. BatGPT-15B-sirius has the following characteristics:

  1. **Supports Contexts Up to 32K Tokens**: BatGPT-15B-sirius uses rotated positional encoding (RoPE) and is pretrained with a sequence length of 2048 tokens. During fine-tuning, it gradually expands to support contexts up to 32K tokens.
  2. **Efficient Pre-training Objectives and Model Architecture**: BatGPT-15B-sirius employs a bidirectional autoregressive pretraining objective to better utilize the training data. It also utilizes the [Multi-Query Attention](http://arxiv.org/abs/1911.02150) technique to reduce inference memory consumption and improve inference speed while maintaining model size.
  3. **Business-friendly Open License**: The source code and weights of BatGPT-15B-sirius are not only available for academic research but also allow free and open-source commercial use, further facilitating the integration of large language models into human daily life.


### 软件依赖

```shell
pip install protobuf transformers cpm_kernels torch>=2.0 streamlit sentencepiece accelerate deepspeed
```

### 简易使用

如下是一个使用 BatGPT-15B-sirius 进行对话的示例:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("MLP-lab/BatGPT-15B-sirius", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("MLP-lab/BatGPT-15B-sirius", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()
history = []
system_prompt = None # 你也可以指定系统提示
response, history = model.chat(tokenizer, "你好", history=history, system_prompt=system_prompt)
print(response)
response, history = model.chat(tokenizer, "介绍一下你自己", history=history, system_prompt=system_prompt)
print(response)
```

Here is an example of a conversation using BatGPT-15B-sirius:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("MLP-lab/BatGPT-15B-sirius", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("MLP-lab/BatGPT-15B-sirius", torch_dtype=torch.float16, trust_remote_code=True).cuda()
model = model.eval()
history = []
system_prompt = None # You can give a system prompt here.
response, history = model.chat(tokenizer, "Hello", history=history, system_prompt=system_prompt)
print(response)
response, history = model.chat(tokenizer, "Please introduce yourself", history=history, system_prompt=system_prompt)
print(response)
```


### 模型详情 (Model Details)


BatGPT-15B-sirius 具体参数和见下表:

|     模型名称       | 隐含层维度  | 层数 | Query头数 | Key/Value头数 |词表大小 | 总参数量 | 训练数据（tokens） | 位置编码 | 最大长度 |
|-------------------------|-------|------------|------------|------------|-----------------|--------|--------|----------------|---------|
| BatGPT-15B-sirius             | 5,632  | 48   | 44    | 2   | 65,536    | 15,030,081,024  | 1 万亿           | [RoPE](https://arxiv.org/abs/2104.09864)    | 32K    |



The specific parameters of BatGPT-15B-sirius are as follows:
|     Model Name       | Hidden Size  | Num Layers | Query Heads | Key/Value Heads |Vocab Size | Total Params | Training Dats（tokens） | Position Embedding | Max Length |
|-------------------------|-------|------------|------------|------------|-----------------|--------|--------|----------------|---------|
| BatGPT-15B-sirius             | 5,632  | 48   | 44    | 2   | 65,536    | 15,030,081,024  | 1 万亿           | [RoPE](https://arxiv.org/abs/2104.09864)    | 32K    |



- **Developed by:** MLP Lab of Wuhan University, Shanghai Jiao Tong University
- **Email**: zcli-charlie@whu.edu.cn, zhaohai@cs.sjtu.edu.cn
- **Language(s) (NLP):** Chinese/English
- **License:** The code in this project is licensed under the Apache 2.0 license, the model weights are licensed under the GNU AGPL 3.0 license. If you intend to use the models included in this project for commercial purposes or public deployment, please email to us to obtain authorization. Commercial usage information will be used for record purposes only and no fees will be charged. 


## 免责声明 (Disclaimers)

BatGPT-15B-sirius 模型的使用应当遵循社会的公序良俗，不能被用于任何危害国家社会安全或违法的活动。另外，我们也要求使用者不要将 BatGPT-15B-sirius 模型用于未经适当安全审查和备案的互联网服务。我们希望所有的使用者都能遵守这个原则，确保科技的发展能在规范和合法的环境下进行。

我们已经尽我们所能，来确保模型训练过程中使用的数据的合规性。然而，尽管我们已经做出了巨大的努力，但由于模型和数据的复杂性，仍有可能存在一些无法预见的问题。如使用本项目所含模型及其修改版本提供服务产生误导性或有害性言论，造成不良影响，由服务提供方负责，与本项目无关。

The use of the BatGPT-15B-sirius model should adhere to societal norms and not be used for any activities that jeopardize national or social security or violate the law. Additionally, we also request users not to use the BatGPT-15B-sirius model for internet services that have not undergone appropriate security review and documentation. We hope that all users will abide by this principle to ensure that technological development occurs in a regulated and legal environment.

We have done our best to ensure the compliance of the data used during the model training process. However, despite our significant efforts, unforeseen issues may still arise due to the complexity of the model and data. If misleading or harmful statements are generated through the use of the models included in this project or their modified versions while providing services, the responsibility lies with the service provider and is not associated with this project.

***

## 引用

如果你觉得我们的工作有帮助的话，请考虑引用我们的BatGPT论文:

If you find our work helpful, please consider citing our BatGPT paper:

```
@article{li2023batgpt,
  title={BatGPT: A Bidirectional Autoregessive Talker from Generative Pre-trained Transformer},
  author={Li, Zuchao and Zhang, Shitou and Zhao, Hai and Yang, Yifei and Yang, Dongjie},
  journal={arXiv preprint arXiv:2307.00360},
  year={2023}
}
```
