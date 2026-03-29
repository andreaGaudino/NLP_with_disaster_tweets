# Project: Natural Language Processing with Disaster Tweets

<!-- TOC -->

- [Project: Natural Language Processing with Disaster Tweets](#project-natural-language-processing-with-disaster-tweets)
    - [Introduction](#introduction)
        - [Abstract](#abstract)
        - [Dataset Description](#dataset-description)
        - [Metrics Description](#metrics-description)
    - [Setup](#setup)
        - [Python Version](#python-version)
        - [Create and Activate the Virtual Environment](#create-and-activate-the-virtual-environment)
        - [Install Dependencies](#install-dependencies)
        - [Download External Data](#download-external-data)
            - [Kaggle competition data](#kaggle-competition-data)
            - [GloVe embeddings required for bilstm_glove.ipynb](#glove-embeddings-required-for-bilstm_gloveipynb)
        - [Run the Notebooks](#run-the-notebooks)
    - [Project Structure](#project-structure)
    - [Results](#results)
    - [References](#references)

<!-- /TOC -->

## Introduction

### Abstract
Social media platforms such as Twitter are widely used during emergencies and natural disasters, and organizations such as disaster relief agencies and news outlets are interested in automatically monitoring these streams. However, distinguishing between tweets that report real disaster events and those that use disaster-related language metaphorically remains a challenging NLP task.

In this project, we develop and evaluate deep learning models for binary classification of disaster-related tweets. The final outcome is a trained system that predicts whether a tweet refers to a real disaster or not. We compare different modeling strategies using standard evaluation metrics. The project is inspired by the Kaggle competition Natural Language Processing with Disaster Tweets.

References: Addison Howard, devrishi, Phil Culliton, and Yufeng Guo. Natural Language Processing with Disaster Tweets. Kaggle, 2019. https://kaggle.com/competitions/nlp-getting-started

### Dataset Description
We utilize a supervised text classification dataset sourced from Kaggle, containing a total of 10,876 tweets (split into 7,613 training and 3,263 testing samples). The dataset poses a classic Natural Language Processing (NLP) binary classification problem. Each data point provides the raw text sequence of the tweet, accompanied by two supplementary metadata features: location and keyword.

Files:
- `data/train.csv`
- `data/test.csv`

### Metrics Description
We evaluate performance using the F1-score. Since this is a binary classification task and the dataset may be imbalanced, F1-score is more appropriate than accuracy. A simple accuracy metric could be misleading, as a model predicting mostly the majority class might still achieve high accuracy while performing poorly on the minority class.

The F1-score balances avoiding false alarms and detecting real disasters. Both missing a real disaster (false negative) and incorrectly flagging a non-disaster tweet (false positive) reduce the usefulness of the system.

## Setup

### Python Version

> **TensorFlow requires Python 3.9–3.12.** Python 3.13+ is not supported.
> See the official guide: https://www.tensorflow.org/install/pip
>
> This project was developed and tested with **Python 3.12.10**.
> Download (Windows 64-bit): https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe

### Create and Activate the Virtual Environment

**Windows (Git Bash):**
```bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
```

**Windows (Command Prompt):**
```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

With the virtual environment active:

```bash
pip install -r requirements.txt
```

> **Mac Apple Silicon (M1/M2/M3):** TensorFlow requires two separate packages.
> After running `pip install -r requirements.txt`, also run:
> ```bash
> pip uninstall tensorflow
> pip install tensorflow-macos tensorflow-metal
> ```
> `tensorflow-metal` enables GPU acceleration via Apple's Metal API (MPS).
> Without it, training runs on CPU only.

To use the notebooks in VS Code, install the Jupyter kernel:
```bash
python -m ipykernel install --user --name=nlp_disaster --display-name "Python (nlp_disaster)"
```

### Download External Data

#### 1. Kaggle competition data
Download `train.csv` and `test.csv` from the competition page and place them in `data/`:
```
https://www.kaggle.com/competitions/nlp-getting-started/data
```

#### 2. GloVe embeddings (required for `bilstm_glove.ipynb`)
The BiLSTM model uses pre-trained GloVe word vectors. The file is 822 MB and is **not tracked by git**.

Download `glove.6B.100d.txt` from Kaggle:
```
https://www.kaggle.com/datasets/danielwillgeorge/glove6b100dtxt
```

Place the file directly in `data/`:
```
data/glove.6B.100d.txt
```

The notebook already points to this path:
```python
GLOVE_PATH = '../data/glove.6B.100d.txt'
```

### Run the Notebooks

Run the notebooks **in order** — each one depends on files produced by the previous one:

| Order | Notebook | Produces |
|-------|----------|----------|
| 1 | `notebook/data_cleaning_augmentation.ipynb` | `data/augmented_train.csv`, `data/test_cleaned.csv` |
| 2 | `notebook/eda.ipynb` | plots in `images/` |
| 3 | `notebook/bilstm_glove.ipynb` | `data/submission_bilstm_glove.csv`, models in `models/` |

To run in VS Code:
1. Open the project folder in VS Code.
2. Open any `.ipynb` file under `notebook/`.
3. Click **Select Kernel** (top right) and choose **Python (nlp_disaster)**.
4. Run cells with `Shift+Enter` or click **Run All**.

## Project Structure
```
NLP_with_disaster_tweets/
├── .venv/                                         # Virtual environment (not tracked by git)
├── data/
│   ├── images/
│   │   ├── bert/
│   │   │   └── bert_training_curves.png           # BERT fine-tuning training curves
│   │   ├── bert_lora/
│   │   │   └── bert_lora_training_curves.png      # BERT + LoRA training curves
│   │   ├── bilstmAndGlove/
│   │   │   ├── bilstm_fold1_curves.png            # BiLSTM training curves per fold
│   │   │   ├── bilstm_fold2_curves.png
│   │   │   ├── bilstm_fold3_curves.png
│   │   │   ├── bilstm_fold4_curves.png
│   │   │   └── bilstm_fold5_curves.png
│   │   └── eda/
│   │       ├── eda_01_target_distribution.png
│   │       ├── eda_02_keyword_analysis.png
│   │       ├── eda_03_meta_features.png
│   │       ├── eda_04_unigrams.png
│   │       ├── eda_04_bigrams.png
│   │       ├── eda_04_trigrams.png
│   │       └── eda_05_train_test_consistency.png
│   ├── train.csv                                  # Original training data
│   ├── test.csv                                   # Original test data
│   ├── glove.6B.100d.txt                          # GloVe embeddings — download separately (822 MB, not on git)
│   ├── augmented_train.csv                        # Augmented + cleaned training data (generated)
│   ├── test_cleaned.csv                           # Cleaned test data (generated)
│   ├── submission_bilstm_glove.csv                # BiLSTM + GloVe predictions (Kaggle: 0.809)
│   ├── submission_bert.csv                        # BERT fine-tuning predictions (Kaggle: 0.839)
│   └── submission_bert_lora.csv                   # BERT + LoRA predictions (Kaggle: 0.827)
├── models/
│   ├── bert/                                      # BERT checkpoints (generated, not on git)
│   ├── bert_lora/                                 # BERT + LoRA checkpoints (generated, not on git)
│   └── bilstm/                                    # BiLSTM checkpoints per fold (generated, not on git)
├── notebook/
│   ├── data_cleaning_augmentation.ipynb           # Data cleaning, mislabeled correction,
│   │                                              # meta-features, back-translation augmentation
│   ├── eda.ipynb                                  # Exploratory data analysis
│   ├── bilstm_glove.ipynb                         # BiLSTM + GloVe baseline (OOF F1: 0.776, Kaggle: 0.809)
│   ├── bert.ipynb                                 # BERT fine-tuning (Val F1: 0.838, Kaggle: 0.839)
│   └── bert_lora.ipynb                            # BERT + LoRA — PEFT (Val F1: 0.827, Kaggle: 0.827)
├── .gitignore
├── readme.md
└── requirements.txt
```

## Results

| Model | Val F1 | Kaggle Public F1 | Trainable params |
|-------|--------|------------------|-----------------|
| BiLSTM + GloVe 100d | 0.776 (OOF, 5-fold CV) | 0.809 | ~1M (100%) |
| BERT fine-tuning (bert-base-uncased) | 0.838 | 0.839 | ~110M (100%) |
| BERT + LoRA (r=8, query+value) | 0.8128 | 0.84554 | ~887K (0.65%) |

## References

- Project 1: https://github.com/nikjohn7/Disaster-Tweets-Kaggle
- Project 2: https://www.kaggle.com/code/bkanupam/disaster-tweets-bilstm-fasttext
- Project 4: https://www.kaggle.com/code/akashkr/tf-keras-tutorial-bi-lstm-glove-gru-part-6
- Project 5: https://github.com/hsiehbocheng/natural-language-processing-with-disaster-tweets/blob/main/README.md
- Project 7: https://www.kaggle.com/code/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert
- Challenge Kaggle: https://www.kaggle.com/competitions/nlp-getting-started/overview
- berttweet: https://www.kaggle.com/code/deepaktripathiuk/eda-data-augment-predict-with-roberta-n-ctf-idf