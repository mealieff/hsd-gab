#!/bin/bash
mkdir -p sing_label_data

# Convert each multilabel dataset to single-label

python3 sing_label_method.py \
  --train_embeddings resampled_data/multiclass_ADASYN_embeddings.npy \
  --train_labels resampled_data/multiclass_ADASYN_labels.npy \
  --output sing_label_data/multiclass_ADASYN_single_label.npy

python3 sing_label_method.py \
  --train_embeddings resampled_data/multiclass_CondensedNearestNeighbour_embeddings.npy \
  --train_labels resampled_data/multiclass_CondensedNearestNeighbour_labels.npy \
  --output sing_label_data/multiclass_CondensedNearestNeighbour_single_label.npy

python3 sing_label_method.py \
  --train_embeddings resampled_data/multiclass_RandomOverSampler_embeddings.npy \
  --train_labels resampled_data/multiclass_RandomOverSampler_labels.npy \
  --output sing_label_data/multiclass_RandomOverSampler_single_label.npy

python3 sing_label_method.py \
  --train_embeddings resampled_data/multiclass_RandomUnderSampler_embeddings.npy \
  --train_labels resampled_data/multiclass_RandomUnderSampler_labels.npy \
  --output sing_label_data/multiclass_RandomUnderSampler_single_label.npy

python3 sing_label_method.py \
  --train_embeddings resampled_data/multiclass_SMOTE_embeddings.npy \
  --train_labels resampled_data/multiclass_SMOTE_labels.npy \
  --output sing_label_data/multiclass_SMOTE_single_label.npy

python3 sing_label_method.py \
  --train_embeddings resampled_data/multiclass_SMOTEENN_embeddings.npy \
  --train_labels resampled_data/multiclass_SMOTEENN_labels.npy \
  --output sing_label_data/multiclass_SMOTEENN_single_label.npy

python3 sing_label_method.py \
  --train_embeddings resampled_data/multiclass_TomekLinks_embeddings.npy \
  --train_labels resampled_data/multiclass_TomekLinks_labels.npy \
  --output sing_label_data/multiclass_TomekLinks_single_label.npy

