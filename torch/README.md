## 项目介绍
![图片](https://user-images.githubusercontent.com/68685718/177709368-0a7de33c-f5d2-47f0-9d91-c1ba671b568b.png)

> 方面级情感分析ABSA作为细粒度情感分析任务近年来受到广泛关注，本项目属于其中的子任务：方面情感分类ASC，目的在于对用户长句评论进行方面级情感极性预测

> 输入：包含一个或多个方面的评论语句；
输出：正向positive，中性neutral，负向negative类别属性

### 主要模型介绍

![图片](https://user-images.githubusercontent.com/68685718/177712610-6344da94-28df-4f7a-a4fa-fa6c1055a177.png)

> 模型主要由两个编码器组成：
**Graph Encoder(左侧)** ：依赖树+RGAT Layer，实际训练中需要多层RGAT Layer才能达到较为满意的效果；
**Contextual Encoder(右侧)** ：分词后的句子+Embedding (BERT/BiLSTM)，Embedding结构中BERT达到的效果更好；

> 使用数据集：MASA、Laptop、Restaurant、Twitter（已包含在作者源码中）
> 参考论文 [IEEE2020] https://ieeexplore.ieee.org/document/9276424
> 参考源码 [RGAT-ABSA] https://github.com/muyeby/RGAT-ABSA

### 主要目标

> 分解目标及时间节点（项目过程中动态调整）
- [ ] 2022.07.07 - 2022.07.09：在服务器上配置环境，并跑通pytorch源码
- [ ] 2022.07.11 - 2022.07.14：将代码迁移到oneflow版本上
- [ ] 2022.07.14 - 2022.07.20：后续Web部署

### 项目负责人和交付时间

项目负责人：王璇
预计完成时间：2022.07.21

## 相关 PR

> 可以完成后再填写，一个项目 issue 可能对应多个 PR

| PR   | 作者 | reviewer | 其它  |      
| ---- | ----- | -------- | ----- | 
|        | 王璇 |  | |      
 
## 对应的云平台 URL

> 可公布后再填写
