# hsd-gab

# README

## Overview

This repository contains code, data, and results for a NLP project investigating the impact of various resampling techniques on hate speech detection in multiclass and multilabel datasets. The research focuses on mitigating class imbalance and label sparsity to improve the prediction of minority classes through data preprocessing methods such as resampling and synthetic example generation. The scripts also facilitate evluation utilizing SVM binary and multiclass classification. 

## Key Files

### Scripts
- **`get-embeddings.py`**  
  Reads in TSV file and extracts BERT-based-uncased embeddings for the dataset in data-preprocessing.

- **`merged-resampler.py`**  
  Implements resampling strategies such as oversampling, undersampling, and synthetic data generation to manage class imbalance.

- **`svm-binary.py`**  
  Trains a binary classifier using Support Vector Machines (SVMs) on resampled data.

- **`svm-multi2.py`**  
  An alternative implementation for multiclass classification with parameter variations.

- **`job-script3.sh`**  
  A shell script for automating training jobs on high-performance computing (HPC) clusters.

### Data Files
- **`ghc_train.tsv`** and **`ghc_test.tsv`**  
  Training and testing datasets from the Gab Hate Speech Corpus: https://osf.io/edua3/. 

- **`train_embeddings.npy`** and **`test_embeddings.npy`**  
  Precomputed embeddings for training and test datasets.

- **`new_train_embeddings.npy`**  
  Augmented embeddings after resampling.

- **`train_labels.npy`** and **`test_labels.npy`**  
  Labels corresponding to training and test data.

- **`new_train_labels.npy`**  
  Updated labels after resampling.

- **`resampled_data`**  
  Directory containing resampled training datasets.

- **`resampledcounts.txt`**  
  Example output of class distributions after resampling.

### Results Files
- **`multiresults.txt`**  
  Results of the multiclass classification experiments.

- **`resampledcounts.txt`**  
  Class counts after applying resampling techniques.

### Usage

### Preprocessing
1. Run `get-embeddings.py` to extract embeddings from the raw dataset.
2. Apply resampling techniques using `merged-resampler.py`.

### Training and Evaluation
1. Use `svm-binary.py`, `svm-multi.py`, or `svm-postembedding.py` to train classifiers.
2. Evaluate performance and analyze results in `multiresults.txt`.

### HPC Automation
Run `job-script3.sh` to automate training jobs on HPC systems.

---
### Appendix: Nested Directory of /resampled_data:

/resampled_data
├── binary
│   ├── RandomUnderSampler
│   │   ├── embeddings.npy
│   │   └── labels.npy
│   ├── CondensedNearestNeighbour
│   │   ├── embeddings.npy
│   │   └── labels.npy
│   ├── TomekLinks
│   │   ├── embeddings.npy
│   │   └── labels.npy
│   ├── SMOTE
│   │   ├── embeddings.npy
│   │   └── labels.npy
│   └── ...
└── multiclass
    ├── RandomUnderSampler
    │   ├── embeddings.npy
    │   └── labels.npy
    ├── CondensedNearestNeighbour
    │   ├── embeddings.npy
    │   └── labels.npy
    ├── TomekLinks
    │   ├── embeddings.npy
    │   └── labels.npy
    └── ...
