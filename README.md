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

After running these commands, you should have all 6 datasets downloaded. You can manually download the datasets from the links in `data/datasets.txt` if one or more of them fail to download.

For the following portions, there are different methods of proceeding: Encoders and LLMs. 

## Encoders

After running both previous commands, all that is required is to start the encoders Docker from the project root (not inside encoders). 

```bash
cd ../
python encoders/orchestrator.py
```

This will run the full training script for the encoders. A Docker image will be created once at the beginning of the runtime, and then the orchestrator will spin up 5 containers over the course of the training. Each container will train one model on all 6 datasets (separately). This will take a long time.
