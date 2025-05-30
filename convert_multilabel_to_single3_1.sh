#!/bin/bash

python3 sing_label_method.py \
  --train_embeddings resampled_data3_1/multiclass_ADASYN_embeddings.npy \
  --train_labels resampled_data3_1/multiclass_ADASYN_labels.npy \
  --output sing_label_data3_1/multiclass_ADASYN_single_label.npy

python3 sing_label_method.py \
  --train_embeddings resampled_data3_1/multiclass_CondensedNearestNeighbour_embeddings.npy \
  --train_labels resampled_data3_1/multiclass_CondensedNearestNeighbour_labels.npy \
  --output sing_label_data3_1/multiclass_CondensedNearestNeighbour_single_label.npy

python3 sing_label_method.py \
  --train_embeddings resampled_data3_1/multiclass_RandomOverSampler_embeddings.npy \
  --train_labels resampled_data3_1/multiclass_RandomOverSampler_labels.npy \
  --output sing_label_data3_1/multiclass_RandomOverSampler_single_label.npy

python3 sing_label_method.py \
  --train_embeddings resampled_data3_1/multiclass_RandomUnderSampler_embeddings.npy \
  --train_labels resampled_data3_1/multiclass_RandomUnderSampler_labels.npy \
  --output sing_label_data3_1/multiclass_RandomUnderSampler_single_label.npy

python3 sing_label_method.py \
  --train_embeddings resampled_data3_1/multiclass_SMOTE_embeddings.npy \
  --train_labels resampled_data3_1/multiclass_SMOTE_labels.npy \
  --output sing_label_data3_1/multiclass_SMOTE_single_label.npy

python3 sing_label_method.py \
  --train_embeddings resampled_data3_1/multiclass_SMOTEENN_embeddings.npy \
  --train_labels resampled_data3_1/multiclass_SMOTEENN_labels.npy \
  --output sing_label_data3_1/multiclass_SMOTEENN_single_label.npy

