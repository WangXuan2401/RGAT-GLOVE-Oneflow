# ABSA-Oneflow

这是 oneflow 版本的代码 RGAT-GloVe ，用于处理方面级情感分析问题

参考论文 "[Investigating Typed Syntactic Dependencies for Targeted Sentiment Classification Using Graph Attention Neural Network](https://arxiv.org/abs/2002.09685)"

参考 pytorch 源码: https://github.com/muyeby/RGAT-ABSA


## Setup

WebIDE 环境要求:

+ 公开镜像 oneflow-0.7.0+torch-1.8.1-cu11.2-cudnn8 

+ 资源配置 6core-42Gi-3090(1Card) 


## Get start

1. 数据准备

   + Restaurants, Laptop, Tweets and MAMS 数据集. (将会提供在`dataset`中)

   + 下载 【 Glove embeddings ](http://nlp.stanford.edu/data/glove.840B.300d.zip)), 
      然后运行

     ```
     awk '{print $1}' glove.840B.300d.txt > glove_words.txt
     ```

     获得 `glove_words.txt`.

2. 建立词汇表vocabulary

   ```
   bash build_vocab.sh
   ```

3. 训练-测试

   进入特定的文件路径下，运行如下命令：

   ``` 
   bash run-MAMS-glove.sh
   ```

4. The saved model and training logs will be stored at directory `saved_models`  



## Results

### GloVe-based Model

|Setting|  Acc  | F1  | Log | Pretrained model |
|  :----:  | :----:  |:---:|  :----:  | :----:  |
| Res14   |  |  |  |  |
| Laptop  |  |  |  |  |
| Tweets  |  |  |  |  |
| MAMS    |  |  |  |  |



## References

```
@ARTICLE{bai21syntax,  
	author={Xuefeng Bai and Pengbo Liu and Yue Zhang},  
	journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing},   
	title={Investigating Typed Syntactic Dependencies for Targeted Sentiment Classification Using Graph Attention Neural Network},   
	year={2021},  
	volume={29}, 
	pages={503-514},  
	doi={10.1109/TASLP.2020.3042009}
}
```



