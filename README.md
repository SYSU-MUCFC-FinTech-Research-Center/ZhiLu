<p align="center" style="display: flex; flex-direction: row; justify-content: center; align-items: center">
🤗 <a href="https://huggingface.co/SYSU-MUCFC-FinTech-Research-Center/ZhiLu-13B-Instruct" target="_blank" style="margin-left: 10px">模型下载</a>
</p>

# News
[2023.10.28] 开源智鹿-13B对话大模型。🤗 [HuggingFace](https://huggingface.co/SYSU-MUCFC-FinTech-Research-Center/ZhiLu-13B-Instruct)

# 目录

- [智鹿](https://github.com/SYSU-MUCFC-FinTech-Research-Center/ZhiLu#智鹿)
  - [介绍]()   
  - [训练细节]()
  - [性能评测]()
  - [对话示例]()
  - [快速使用]()
  - [模型训练]()

# 智鹿

## 介绍
智鹿是一个基于中文Alpaca2-13B进行二次训练的金融大模型，我们使用大量中英文语料进行增量预训练，同时使用高质量指令数据进行对齐。模型训练的目标是在保持通用能力的前提下，显著提升金融领域的能力。

## 训练细节

- 我们收集了各类数据，包括上市公司公告、财经新闻、上市公司年度报告、新闻、金融资讯、社区问答、维基百科等高质量数据。

- 模型训练的总token数为`14.69B`，通用语料与金融预料比例约为2:1，中英配比约为2:1。

- 使用LoRA进行高效训练（含emb/lm-head），部分超参数设置如下：

```sh
lr=2e-4
lora_rank=64
lora_alpha=128
lora_trainable="q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
modules_to_save="embed_tokens,lm_head"
lora_dropout=0.05
per_device_train_batch_size=64
per_device_eval_batch_size=64
gradient_accumulation_steps=1
```

- 使用FlashAttention-2技术进行加速训练。

## 性能评测

我们对模型的评测分为通用领域和金融领域两方面的评测。

在通用能力评测中，我们选择当下主流的两类客观评测基准：

- [C-Eval](https://cevalbenchmark.com/index.html#home) 是一个全面的中文基础模型评估基准。它包含了13948个多项选择题，涵盖了52个不同的学科和四个难度级别
- [CMMLU](https://github.com/haonan-li/CMMLU) 是一个综合性的中文评估基准，专门用于评估语言模型在中文语境下的知识和推理能力。CMMLU涵盖了从基础学科到高级专业水平的67个主题。

金融能力评测的基准：

- [FinEval ](https://github.com/SUFE-AIFLM-Lab/FinEval/)是一个专门为LLMs中的金融领域知识而设计的基准测试，涵盖金融、经济、会计和证书等领域。它包括4,661个问题，涵盖了34个不同的学术科目。（需要声明的是，由于FinEval并未公开test集的answer，也未开放评测接口，因此本文在FinEval上的评测上使用了val集）

与底座模型Alpaca2-13B的评测结果对比如下：

| 模型   | CEVAL   | CMMLU | FinEval |
| ------ | ------ | ------ | ------ |
| 智鹿 | 50.1 | 57.14 |58.69 |
| Alpaca2-13B | 43.6 | 44.48 |45.26 |

- 在通用领域评测中，智鹿对比Alpaca2-13B，保留了基本的通用能力，Ceval得分些许提高，CMMLU得分显著提高。
- 在金融领域评测FInEval中，智鹿的性能显著提高。

## 对话示例

### 金融

```
问题：怎么炒股才能稳赚不赔？
智鹿：股市投资是有风险的，无法保证稳赚不赔。以下是一些建议来帮助您在股票市场中实现更稳定的投资回报：1. 分散投资组合：将资金分配到不同的行业、公司和资产类别上，以降低特定投资的风险。这样即使某些投资表现不佳，其他投资也可能弥补损失。2. 长期投资策略：选择具备稳定增长潜力的优质企业，并持有它们的股票较长时间。长期投资可以减少短期市场波动对投资的影响，并获得潜在的资本增值和股息收入。3. 定期重新评估投资计划：根据市场情况和个人目标，定期检查和调整投资组合。这可以帮助您及时做出必要的调整，以适应市场的变化并保持符合您的目标。4. 控制情绪：避免受到市场噪音和情绪影响的投资决策。不要盲目跟风或恐慌出局，而是根据自己的研究和分析制定合理的投资计划。5. 学习和继续教育：了解基本的投资原理和技术分析工具，不断提升自己的知识水平。这有助于更好地理解市场趋势和投资机会，从而作出明智的决策。请注意，以上建议仅供参考，并不能保证投资成功。在进行任何投资之前，请务必咨询专业金融顾问的意见，以便为您的具体情况提供个性化的建议。

问题：什么是A股？
智鹿：简单来说，就是中国内地的股票市场。它包括上海证券交易所和深圳证券交易所。A股市场起源于1990年，是中国股票市场的主要组成部分。A股市场的特点：1. 开放性：A股市场对外开放程度较高，允许国内外投资者买卖股票。2. 风险较高：与发达国家相比，A股市场波动较大，投资风险相对较高。3. 监管严格：中国证监会负责监督和管理A股市场，并采取措施保护投资者利益。4. 长期投资机会多：尽管存在风险，但A股市场也提供了丰富的长期投资机会。对于投资者而言，了解A股市场的运作机制、相关规则和最新动态非常重要。这有助于他们做出明智的投资决策，并在A股市场中获取收益。
```

### 通用领域


```
问题：潮汕地区有什么特产？
智鹿：1.牛肉：潮汕牛肉以肉质细嫩、多汁，且不腥臊腥著称。2.虾螺贝类：汕头地区的海产品丰富，包括各种海鲜如鲍鱼、海参、石斑鱼等。3.柑橘：潮州的柑橘以其独特的清香和丰富的营养价值而闻名。4.姜撞子：是潮州特色小吃之一，将猪肉、鸡蛋、木姜黄等馅料包裹在糯米粉里制成。5.牛肉炒米：用牛肉和糯米制作而成，口感软糯，深受人们喜爱。6.沙士：是一种传统的甜点，由糯米粉制成，并加入花生碎、芝麻等配料。7.牛肉丸：潮汕地区的牛肉丸因其Q弹口感而著名，通常配上辣椒酱或特制的汤汁食用。8.大龙粿：一种传统潮州糕点，采用粘糯米制成，内陷有豆沙或红绿丝等。9.牛肉火锅：潮汕地区有许多著名的牛肉火锅店，其特点是使用高质量的牛肉和独特的调料。10.冬至面：在冬至日，许多家庭会准备热气腾腾的冬至面，其中包含着多种肉类和蔬菜。以上只是一些常见的潮汕美食和特产，还有更多值得探索的地方和美食等待你去发现。

问题：晚上睡不着怎么办？
智鹿：1. 做点轻松的事，如阅读、写字等。2. 听音乐，放松身心。3. 数羊或数数，转移注意力。4. 洗个澡，让自己放松下来。5. 出去走走，呼吸新鲜空气。6. 与朋友聊天，分享烦恼并寻求帮助。7. 写日记，记录今天发生的事情和感受。8. 运动运动，释放压力和紧张情绪。9. 冥想或练习深呼吸，平静自己的心神。10. 看喜剧片，暂时忘记忧愁。

问题：介绍一下中国象棋的规则。
智鹿：1. 走法：双方各自有15块棋子，包括王、车、兵、相、马和炮。每种单位都有特定的走法，例如，王只能前进一步或退后到安全的位置，而其他单位则可以根据具体情况选择不同的走法。2. 控制区域：每个玩家拥有自己的半盘，称为“阵营”。在战斗中，可以将对方的单位推回自己阵营之外的区域内，从而控制该区域。3. 攻击方式：不同类型的单位具有不同的攻击能力。例如，车可以向前推进一个空格，同时可以把敌人的单位移开；兵则可以从初始位置出发，沿着直线向对手发起进攻。4. 防御策略：为了保护己方单位免受攻击，玩家可以通过调整布局来制定有效的防御策略。这可能涉及到隐藏单位、利用障碍物（如山石）以及进行交换等手段。5. 中国象棋的历史和文化价值：中国象棋起源于中国古代，有着悠久的历史。它不仅作为一种智力游戏受到人们喜爱，还蕴含着丰富的哲学思想和道德修养。通过学习和参与中国象棋比赛，人们可以增强团队合作意识、培养决断力和战略思维能力。
```

## 使用方式

我们已经把智鹿的模型权重发布在HuggingFace，由于我们的底座模型是Alpaca2，而Alpaca2是基于Llama2，因此可以使用Llama2-hf的调用方式，如下：

我们提供两种模型调用方式：`模型合并后直接加载`和`底座模型、LoRA模块分开加载`

温馨提醒：

1、使用智鹿需要约30GB显存。

2、prompt的结尾尽量加上标点符号，否则模型可能会续写prompt。

### 环境安装
使用 pip 安装依赖：
```
pip install -r requirements.txt
```
其中 `transformers` 库版本推荐为 `4.30`以上，`torch` 推荐使用 2.0 及以上的版本，以获得最佳的推理性能。

### 合并后使用

需将Alpaca-2-13b与我们提供的LoRA模块进行合并后，才能使用，合并脚本：`\scripts\merge.sh`

模型使用时只需提供合并后的地址：

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
model_name_or_path = ""
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=False, legacy=True)
model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16,device_map="auto")
inputs = tokenizer("什么是A股？", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
outputs = tokenizer.decode(outputs.cpu()[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(outputs)
```

### 加载LoRA模块

需提供Alpaca-2-13b和LoRA模块的地址：

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
model_name_or_path = ""
peft_model_path = ""
tokenizer = LlamaTokenizer.from_pretrained(model_name_or_path, use_fast=False, legacy=True)
model = LlamaForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16,device_map="auto")
if peft_model_path is not None:
    model = PeftModel.from_pretrained(
            model,
            peft_model_path,
            torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float32
        ),
    )
inputs = tokenizer("什么是A股？", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=64, repetition_penalty=1.1)
outputs = tokenizer.decode(outputs.cpu()[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
print(outputs)
```

# To Do List

1.性能强化，应用DPO，发布-chat版本

2.对标消费金融领域的实际场景进行性能增强

# FAQ

问题1：模型在预训练还是在指令微调阶段进行LoRA训练？

本项目在预训练和指令微调阶段均使用LoRA进行高效训练。

问题2：为什么对模型进行LoRA训练而非全量参数训练？

考虑到训练成本和效率等因素，我们选择在Alpaca-2的基础上使用LoRA进行训练（embedding/lm_head全量参与训练）。

问题3：为什么选择在Alpaca-2的基础上进行训练，而非Llama-2？

Llama-2虽然已具备一定的中文理解能力，但在生成中文文本时仍然会夹杂英文，且中文词表大小有限，需要进行进一步的中文能力扩展，Alpaca-2在该方面已经做出了出色的工作。

# 致谢

本项目主要基于Alpaca2开源项目进行二次开发，在此对相关项目和研究开发人员表示感谢。

# 免责声明与许可协议
本仓库的代码依照[Apache-2.0](https://github.com/SYSU-MUCFC-FinTech-Research-Center/ZhiLu/blob/main/LICENSE)协议开源。模型权重对学术研究完全开放；使用者也可通过[申请表]()申请、经同意并发放商用授权证书后免费用于商业用途。
尽管我们在模型训练过程中尽力确保数据的合规性和准确性，但由于模型受概率随机性因素影响及易被误导，无法保证输出内容的准确性。因此，使用者在使用本模型及其生成的内容时，应自行审慎识别后作出独立判断，必要时应征询专业人士意见，并由使用者承担使用风险。使用者也不得将本模型用于任何可能给国家和社会带来危害的用途以及用于任何未经过安全评估和备案的服务。我们不承担开源模型及其生成的内容导致的安全风险、知识产权风险、舆情风险或发生任何模型被误导、滥用、不当利用及传播而产生的风险和责任。

# 总结

我们鼓励使用者在相关工作中引用智鹿，以促进知识的共享和交流，并为中文金融对话系统的不断发展贡献力量。
智鹿的发布旨在为金融领域的应用和研究提供有力支持，为中文金融对话系统的进步做出积极贡献。我们期待见证更多的创新和应用案例，以提升金融服务和用户体验，同时也推动人工智能技术在金融领域的蓬勃发展。通过合作和分享，我们可以共同推动这一领域的发展，为社会和行业带来更多的好处。
