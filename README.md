# PointLoRA
> Song Wang, Xiaolu Liu, Lingdong Kong, Jianyun Xu, Chunyong Hu, Gongfan Fang, Wentong Li, Jianke Zhu, Xinchao Wang

This is the official implementation of **PointLoRA: Low-Rank Adaptation with Token Selection for Point Cloud Learning** (CVPR 2025)  [[Paper](https://arxiv.org/pdf/2504.16023)] [[Video]()].

## Abstract
Self-supervised representation learning for point cloud has demonstrated effectiveness in improving pre-trained model performance across diverse tasks. However, as pre-trained models grow in complexity, fully fine-tuning them for downstream applications demands substantial computational and storage resources. Parameter-efficient fine-tuning (PEFT) methods offer a promising solution to mitigate these resource requirements, yet most current approaches rely on complex adapter and prompt mechanisms that increase tunable parameters. In this paper, we propose PointLoRA, a simple yet effective method that combines low-rank adaptation (LoRA) with multi-scale token selection to efficiently fine-tune point cloud models. Our approach embeds LoRA layers within the most parameter-intensive components of point cloud transformers, reducing the need for tunable parameters while enhancing global feature capture. Additionally, multi-scale token selection extracts critical local information to serve as prompts for downstream fine-tuning, effectively complementing the global context captured by LoRA. The experimental results across various pre-trained models and three challenging public datasets demonstrate that our approach achieves competitive performance with only 3.43% of the trainable parameters, making it highly effective for resource-constrained applications. 

<p align="center"> <a><img src="fig/framework.png" width="90%"></a> </p>

## Acknowledgement
We gratefully acknowledge the contributions of various open-source projects that supported this work:
[DAPT](https://github.com/LMD0311/DAPT), [PPT](https://github.com/zsc000722/PPT), [Point-PEFT](https://github.com/Ivan-Tang-3D/Point-PEFT).
