# Multimodal Medical LLama (MLLM)
A fine tuned LLama model that accepts multimodal data as input and gives response as text. The model accepts data in text, audio and time series data (ECG Data). 


## Introduction
Multimodal language models have gained significant attention in recent years due to their ability to process diverse sources of information, such as text, speech, and time series data. This repository contains an implementation of an MLLM that leverages the power of pre-trained language models to perform various tasks across multiple modalities.

## Installation
- Clone the repository:
```git clone https://github.com/your-username/multimodal-language-model.git```
```cd multimodal-language-model```
- Create a virtual environment (optional but recommended):
```python -m venv venv```
```source venv/bin/activate```
- Install the required dependencies:
```pip install -r requirements.txt```

## Training 
- For training the CNN architecture for Multilabel data classification (ECG). 
  - Download the data from the website [Kaggle](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset) and place it in ```time_series/data/```
  - Run ```python time_series/train.py```
- For fine-tuning the LLama Model (GPU Required). 
  - Download the instruction tuned dataset and place it in ```llm/dataset/```
  - Run ```python llm/training.py```

## Inferencing 
- For inferencing the model run ```python inference.py```



