# Project: Natural Language Processing with Disaster Tweets

<!-- TOC -->

- [Project: Natural Language Processing with Disaster Tweets](#project-natural-language-processing-with-disaster-tweets)
    - [Introduction](#introduction)
        - [Abstract](#abstract)
        - [Dataset Description](#dataset-description)
        - [Metrics Description](#metrics-description)

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
