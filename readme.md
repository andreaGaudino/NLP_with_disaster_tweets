# Project: Natural Language Processing with Disaster Tweets

<!-- TOC -->

- [Project: Natural Language Processing with Disaster Tweets](#project-natural-language-processing-with-disaster-tweets)
    - [Introduction](#introduction)
        - [Abstract](#abstract)
        - [Dataset Description](#dataset-description)
        - [Metrics Description](#metrics-description)
    - [Setup](#setup)
        - [Create and Activate the Virtual Environment](#create-and-activate-the-virtual-environment)
        - [Install Dependencies](#install-dependencies)
        - [Run the Notebooks](#run-the-notebooks)
    - [Project Structure](#project-structure)

<!-- /TOC -->

## Introduction

### Abstract
Social media platforms such as Twitter are widely used during emergencies and natural disasters, and organizations such as disaster relief agencies and news outlets are interested in automatically monitoring these streams. However, distinguishing between tweets that report real disaster events and those that use disaster-related language metaphorically remains a challenging NLP task.
In this project, we will develop and evaluate deep learning models for binary classification of disaster-related tweets. The final outcome is a trained system that predicts whether a tweet refers to a real disaster or not. We will compare different modeling strategies using standard evaluation metrics. The project is inspired by the Kaggle competition Natural Language Processing with Disaster Tweets.
References: Addison Howard, devrishi, Phil Culliton, and Yufeng Guo. Natural Language Processing with Disaster Tweets. Kaggle, 2019. https://kaggle.com/competitions/nlp-getting-started

### Dataset Description
We utilize a supervised text classification dataset sourced from Kaggle, containing a total of 10,876 tweets (split into 7,613 training and 3,263 testing samples). The dataset poses a classic Natural Language Processing (NLP) binary classification problem. Each data point provides the raw text sequence of the tweet, accompanied by two supplementary metadata features: location and keyword. A core technical challenge of this project will involve not only parsing the noisy, unstructured text data, but also engineering the potentially sparse metadata to improve the classifier's predictive performance and semantic understanding.
Files
/train.csv
/test.csv

### Metrics Description
We evaluate performance using the F1-score. Since this is a binary classification task and the dataset may be imbalanced, F1-score is more appropriate than accuracy. A simple accuracy metric could be misleading, as a model predicting mostly the majority class might still achieve high accuracy while performing poorly on the minority class.

The F1-score balances avoiding false alarms and detecting real disasters. This is important in our context, since both missing a real disaster (false negative) and incorrectly flagging a non-disaster tweet (false positive) can negatively impact the usefulness of the system.

## Setup
### Create and Activate the Virtual Environment
From the root of the project, create a virtual environment and activate it.

**Windows (Git Bash / Command Prompt):**
```bash
python -m venv .venv
source .venv/Scripts/activate   # Git Bash
# or
.venv\Scripts\activate          # Command Prompt
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies
With the virtual environment active, install all required packages:
```bash
pip install -r requirements.txt
```

To use the notebooks in VS Code, also install the Jupyter kernel:
```bash
pip install ipykernel
python -m ipykernel install --user --name=nlp_disaster --display-name "Python (nlp_disaster)"
```

### Run the Notebooks
1. Open the project folder in VS Code.
2. Open any `.ipynb` file under `notebook/`.
3. Click **Select Kernel** (top right) and choose **Python (nlp_disaster)**.
4. Run cells with `Shift+Enter` or click **Run All**.

## Project Structure
```
NLP_with_disaster_tweets/
├── .venv/                              # Virtual environment (not tracked by git)
├── data/
│   ├── train.csv                       # Original training data
│   ├── test.csv                        # Original test data
│   ├── augmented_train.csv             # Augmented + cleaned training data (generated)
│   └── test_cleaned.csv                # Cleaned test data (generated)
├── images/
│   ├── eda_01_target_distribution.png  # Target class balance (57% / 43%)
│   ├── eda_02_keyword_analysis.png     # Disaster rate per keyword
│   ├── eda_03_meta_features.png        # Meta-features distribution by class
│   ├── eda_04_unigrams.png             # Top 20 unigrams per class
│   ├── eda_04_bigrams.png              # Top 20 bigrams per class
│   ├── eda_04_trigrams.png             # Top 20 trigrams per class
│   └── eda_05_train_test_consistency.png # Train/test distribution comparison
├── notebook/
│   ├── data_cleaning_augmentation.ipynb  # Data cleaning, mislabeled correction,
│   │                                     # meta-features, back-translation augmentation
│   └── eda.ipynb                         # Exploratory data analysis
├── .gitignore
├── readme.md
└── requirements.txt
```