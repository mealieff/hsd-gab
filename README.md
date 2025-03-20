# hsd-gab

# README

## Overview

This repository contains code, data, and results for an NLP project investigating the impact of various resampling techniques on hate speech detection in multiclass and multilabel datasets. The research focuses on mitigating class imbalance and label sparsity to improve the prediction of minority classes through data preprocessing methods such as resampling and synthetic example generation. The scripts also facilitate evaluation utilizing SVM binary and multiclass classification. 

## Key Files

### Scripts
- **`get-embeddings.py`**  
  Reads in TSV file and extracts BERT-based-uncased embeddings for the dataset in data-preprocessing.

- **`merged-resampler.py`**  
  Implements resampling strategies such as oversampling, undersampling, and synthetic data generation to manage class imbalance.

- **`svm-binary.py`**  
  Trains a binary classifier using Support Vector Machines (SVMs) on resampled data.

- **`svm-multi.py`**  
  Trains a multi-class classifier using Support Vector Machines (SVMs) resampled data.

- **`job-script3.sh`**  
  A shell script for automating training jobs on high-performance computing (HPC) clusters.

### Data Files
- **`ghc_train.tsv`** and **`ghc_test.tsv`**  
  Training and testing datasets from the Gab Hate Speech Corpus: https://osf.io/edua3/. 

- **`train_embeddings.npy`** and **`test_embeddings.npy`**  
  Precomputed embeddings for training and test datasets.

- **`train_labels.npy`** and **`test_labels.npy`**  
  Labels corresponding to training and test data.

- **`resampled_data`**  
  Directory containing resampled training datasets.


### Results Files
- **`resampled_data_binary.txt`**  
  Results of the binary resampling.

- **`resampled_data_multi.txt`**  
  Results of the multi resampling. 

- **`svm_binary.txt`**  
  Results of the classification on binary labels. 
  
- **`svm_multi.txt`**  
  Results of the classification on multi labels.

- **`svm-binary-baseline.txt`**  and - **`svm-multi-baseline.txt`** 
  Results of the classification on binary and multi labels without any resampling.

- **`svm-multi-metrics-4lables.txt`**  and - **`svm-multi-metrics-8lables.txt`**
  Results of the classification on binary and multi labels using confidence scores. 



### Usage

### Preprocessing
1. Run `get-embeddings.py` to extract embeddings from the raw dataset.
2. Apply resampling techniques using `merged-resampler.py`.

### Training and Evaluation
1. Use `svm-binary.py`, `svm-multi.py`, or `svm-postembedding.py` to train classifiers.
2. Evaluate performance and analyze results in `results-hsd-gab`.

---
### Appendix: Nested Directory of /resampled_data:

/resampled_data</br>
├── binary</br>
│   ├── RandomUnderSampler</br>
│   │   ├── embeddings.npy</br>
│   │   └── labels.npy</br>
│   ├── CondensedNearestNeighbour</br>
│   │   ├── embeddings.npy</br>
│   │   └── labels.npy</br>
│   ├── TomekLinks</br>
│   │   ├── embeddings.npy</br>
│   │   └── labels.npy</br>
│   ├── SMOTE</br>
│   │   ├── embeddings.npy</br>
│   │   └── labels.npy</br>
│   └── ...</br>
└── multiclass</br>
    ├── RandomUnderSampler</br>
    │   ├── embeddings.npy</br>
    │   └── labels.npy</br>
    ├── CondensedNearestNeighbour</br>
    │   ├── embeddings.npy</br>
    │   └── labels.npy</br>
    ├── TomekLinks</br>
    │   ├── embeddings.npy</br>
    │   └── labels.npy</br>
    └── ...</br>
