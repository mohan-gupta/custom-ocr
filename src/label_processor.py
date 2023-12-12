import joblib

import numpy as np
import pandas as pd

import config

def label_to_map(labels:pd.Series):
    """
    Function to convert label strings to list of characters
    """
    label_lst = labels.apply(lambda x: list(x)).to_list()
    
    flatten_labels=[]
    for label in label_lst:
        flatten_labels.extend(label)
        
    char_set = np.sort(np.unique(flatten_labels))
    
    idx_to_char = dict(enumerate(char_set))
    char_to_idx = {char:idx for idx, char in idx_to_char.items()}
    
    return idx_to_char, char_to_idx

def transform_label(char_to_idx, labels):
    trnsfrmd_labels = []
    for label in labels:
        num_label = []
        for char in label:
            idx = char_to_idx[char]+1
            num_label.append(idx)
        
        trnsfrmd_labels.append(num_label)
        
    return trnsfrmd_labels

def encode_labels(train_labels, valid_labels, test_labels=None, load=False):
    if load is False:
        idx_to_char, char_to_idx = label_to_map(train_labels)
        joblib.dump((idx_to_char, char_to_idx), config.LABEL_MAPPER)
        print(f"Num Classes: {len(idx_to_char)}")
    else:
        idx_to_char, char_to_idx = joblib.load(config.LABEL_MAPPER)
        
    trnsfrmd_train_labels = transform_label(char_to_idx, train_labels.apply(lambda x: list(x)))
    trnsfrmd_valid_labels = transform_label(char_to_idx, valid_labels.apply(lambda x: list(x)))

    if test_labels is not None:
        trnsfrmd_test_labels = transform_label(char_to_idx, test_labels.apply(lambda x: list(x)))

        return {
            "train_labels": trnsfrmd_train_labels,
            "valid_labels": trnsfrmd_valid_labels,
            "test_labels": trnsfrmd_test_labels,
            "num_classes": len(idx_to_char)
        }
    
    return {
        "train_labels": trnsfrmd_train_labels,
        "valid_labels": trnsfrmd_valid_labels,
        "num_classes": len(idx_to_char)
    }