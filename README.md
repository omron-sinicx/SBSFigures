# SBS Figures: Pre-training Figure QA from Stage-by-Stage Synthesized Images
<a href='https://arxiv.org/abs/2412.17606'><img src='https://img.shields.io/badge/ArXiv-PDF-red'></a> &nbsp; 
<a href='https://omron-sinicx.github.io/SBSFiguresPage/'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp; 

The official PyTorch implementation for the following paper:
> [**SBS Figures: Pre-training Figure QA from Stage-by-Stage Synthesized Images**](https://arxiv.org/abs/2407.13555),  
> [Risa Shionoda](https://sites.google.com/view/risashinoda/home), [Kuniaki Saito](https://ksaito-ut.github.io/),[Shohei Tanaka](https://shohei-ta-ds7.github.io/),[Tosho Hirasawa](https://toshohirasawa.github.io/),[Yoshitaka Ushiku](https://yoshitakaushiku.net/index.html)    
> *The AAAI-25 Workshop on Document Understanding and Intelligence*

### TL;DR: Introducing the SBSFigures generation pipeline—effortlessly create figures with diverse topics, varied appearances, and precise QAs using a single bash script.

## Dataset
You can download our SBS Figures dataset (1M figures, 4.2M QA pairs) from Hugging Face: [Hugging Face Dataset](https://huggingface.co/datasets/omron-sinicx/sbsfigures)

## Dataset Generation Pipeline
You can also create SBSFigures using our generation pipeline.

The generation pipeline consists of the following Python scripts.
1. data_topic.py : Create figure topic
2. json_make.py:  Create JSON files representing data point
3. add_color.py: Add color information to the JSON files
4. create_chart.py: Create chart png files 
5. qa.py: Create qa pairs using data points

You can create SBSFigures by GPT with the following command:
```
cd data_gen/gpt
bash crete_sbsfigures.sh
```
You have to modify config.yaml and write your openai API key.
Be careful, this GPT-based generation cost money, and try the lower nuber of Figures. (Initially, we set 15 figures generation attempts per figure type.)

## Model
We release four models through Hugging Face.

| Task  | Model | Checkpoint Path |
| ------| ------- | ------------- |
| Pretrained  | Donut| [omron-sinicx/sbsfigures-pretrain-donut](https://huggingface.co/omron-sinicx/sbsfigures-pretrain-donut)  |
| Fine-tuned (ChartQA) | Donut | [omron-sinicx/sbsfigures-chartqa-donut](https://huggingface.co/omron-sinicx/sbsfigures-chartqa-donut)  |
| Pretrained  | Pix2Struct| [omron-sinicx/sbsfigures-pretrain-pix2struct](https://huggingface.co/omron-sinicx/sbsfigures-pretrain-pix2struct)  |
| Fine-tuned (ChartQA) |Pix2Struct| [omron-sinicx/sbsfigures-chartqa-pix2struct](https://huggingface.co/omron-sinicx/sbsfigures-chartqa-pix2struct)  |

## Setup
docker build :
```
docker build -t sbsfigures:latest -f SBSFigures/Dockerfile SBSFigures
```
docker run :
```
docker run -it --rm -v SBSFigures:/app SBSFigures:latest /bin/bash
```

## Pre-training Code
Donut : 
```
cd donut
bash pre-train_sbsfigures.sh
```
Pix2Struct : 
```
cd pix2struct
bash pre-train_sbsfigures.sh
```

## Fine-tuning Code
Donut : 
```
cd donut
bash finetune_chartqa.sh
```
Pix2Struct : 
```
cd pix2struct
bash finetune_chartqa.sh
```
For the fine-tuning, we borrow some code from [UniChart](https://github.com/vis-nlp/UniChart).

## Tips
- **Customize fonts**:  
  Edit `data_gen/fpt/font.txt` to add or remove fonts based on your environment for chart creation.

- **Create domain-specific figures**:  
  Modify the prompt in `data_gen/gpt/data_topic.py` to generate figures tailored to a specific domain.

- **Add a new figure type**:  
  To add a figure type (e.g., a leather chart), define the `code_format` in `data_gen/code_format`, specify its JSON style, and add examples to `data_gen/example/data_point/(your new figure type)`.

# Citation
If you find our work useful for your research, please consider citing our paper:

```bibtex
@article{shinoda2024sbsfigurespretrainingfigure,
title={SBS Figures: Pre-training Figure QA from Stage-by-Stage Synthesized Images}, 
author={Risa Shinoda and Kuniaki Saito and Shohei Tanaka and Tosho Hirasawa and Yoshitaka Ushiku},
year={2024},
journal={arXiv preprint arXiv:2412.17606},
url={https://arxiv.org/abs/2412.17606}
}
```
