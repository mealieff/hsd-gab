# hsd-gab

# README

## Overview

This repository contains code, data, and results for an NLP project investigating the impact of various resampling techniques on hate speech detection in multiclass and multilabel datasets. The research focuses on mitigating class imbalance and label sparsity to improve the prediction of minority classes through data preprocessing methods such as resampling and synthetic example generation. The scripts also facilitate evaluation utilizing SVM binary and multiclass classification. 

## Key Files
### Resampling Scripts
- **`merged-resampler.py`**  
  Applies various resampling techniques (e.g., RandomUnderSampler, SMOTE, TomekLinks, CondensedNearestNeighbour) to address class imbalance in both binary and multiclass settings. Outputs resampled datasets as NPY files.

- **`svm-postembedding.py`**  
  Performs SVM classification using post-embedding features, allowing for additional evaluation scenarios.

### Scripts
- **`get-embeddings.py`**  
  Reads in TSV file and extracts BERT-based-uncased embeddings for the dataset in data-preprocessing.

- **`main.py`**
  Reads in NPY arrays and trains/outputs svm classification results. See argument parser details for adjusting parameters for multilabel classification. Usually run in batches using ```run_all_evaluations.sh```

  This script is used for single label methods. Can use argument parsing to set parameters and utilize dev set to adjust parameters. Needs debugging.  
  
### Data Files
- **`ghc_train.tsv`** and **`ghc_test.tsv`**  
  Training and testing datasets from the Gab Hate Speech Corpus: https://osf.io/edua3/. 

- **`resampled_data`**  
  Directory containing resampled training datasets.

### Usage

### Preprocessing
1. Run `get-embeddings.py` to extract embeddings from the raw dataset.
2. Apply resampling techniques using `merged-resampler.py`.

### Training and Evaluation
1. Use `main.py` to train and test SVM classifier.
2. Evaluate performance and analyze results in `results-hsd-gab`.

### Thresholding label training.
1. Use `main.py` to train and test SVM classifier. See argument parsing logic to adjust the settings.
2. Run using a variety of confidence thresholds using `run_all_evaluations.sh` or `run_all_single_label.sh` adjusting the logic for your directory setup as necessary. 
3. Alternatively, use the dev set to to set the threshold for the test set. 

