# Sarcasm Detection Project

Repo for Sarcasm Detection project for NLP.

Testing encoders against LLMs in sarcasm detection

## Encoders

We will be using 5 different encoder-only transformer models.

### 1. BERT


### 2. RoBERTa


### 3. DeBERTa


### 4. DistilBERT


### 5. ELECTRA


## LLM's

We will be comparing the outputs of 4 LLM's on zero-shot and few-shot training

### 1. ChatGPT


### 2. Claude


### 3. Gemini


### 4. Grok


# Setup

To get started (using either encoders or LLMs), you will first need the data. All data used in this project is open source and can be downloaded for free, without any accounts or payments needed. Simply run the `download.py` script, and once that finishes you can run the `preprocessing.py` script. This will download the data from their respective sources, then break apart the downloaded data into it's necessary components so that they can be used for fine-tuning/testing.

```bash
cd data
python download.py
python preprocessing.py
```
