"""Data loading."""
import nltk

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split, ConcatDataset
import numpy as np
import random

from nltk.corpus import wordnet
from nltk.corpus import stopwords
from utils import set_seed
from config import BaselineConfig
from preprocessing import Augmentor , Chr_encoder


nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")




class Chr_dataset(Dataset):
    def __init__(self, data, encoder, augmentor, augmentation_flag=False):
        self.text = data['text']
        self.label = data['label']
        self.encoder = encoder
        self.augmentation_flag = augmentation_flag
        self.augmentor = augmentor

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.label[idx]

        if self.augmentation_flag:
            text = self.augmentor.augment_text(text)

        X = self.encoder.encode(text)
        y = torch.tensor(label)
        return X, y


def get_dataloader(config: BaselineConfig, split='train'):
    """Create dataloader."""
    set_seed(config.seed)

    dataset = load_dataset(config.dataset_name)
    splited_train = dataset['train'].train_test_split(train_size=0.8, seed=config.seed)
    train_d = splited_train['train']
    valid_d = splited_train['test']
    test_d = dataset['test']

    enc = Chr_encoder(max_length=config.text_len, alpha_length=config.alpha_len)
    aug = Augmentor(config.p, config.q)

    if split == "train":
        train_dataset_aug = Chr_dataset(train_d, enc, aug, True)
        train_dataset_unaug = Chr_dataset(train_d, enc, aug, False)
        train_dataset = ConcatDataset([train_dataset_aug, train_dataset_unaug])

        final_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    elif split == "val":
        valid_dataset = Chr_dataset(valid_d, enc, aug, False)
        final_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    else:
        test_dataset = Chr_dataset(test_d, enc, aug, False)
        final_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return final_dataloader
