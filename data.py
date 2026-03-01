"""Data loading."""
import nltk
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split, ConcatDataset
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from utils import set_seed
from config import BaselineConfig

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
stop_word = set(stopwords.words("english"))


class Augmentor:
    def __init__(self, p=0.5, q=0.5):
        self.p = p
        self.q = q

    def get_syns(self, word: str):
        synonyms = set()

        if word.isalnum():

            for syn in wordnet.synsets(word):
                for lemma in syn.lemmas():
                    root = lemma.name().lower()
                    if (root != word):
                        synonyms.add(root)

        return list(synonyms)

    def augment_text(self, text):
        words = nltk.word_tokenize(text)
        augmentable = []

        for idx, word in enumerate(words):
            if len(self.get_syns(word)):
                augmentable.append(idx)

        num_augmentable = len(augmentable)

        r = np.random.geometric(p=self.p)  # number of words to be augmented

        if (not num_augmentable) or (r == 0):
            return text

        if r > num_augmentable:
            r = num_augmentable

        idx_to_augment = random.sample(augmentable, r)

        for idx in idx_to_augment:
            synonyms = self.get_syns(words[idx])
            m = np.random.geometric(p=self.q)
            m = m - 1
            if m >= len(synonyms):
                m = len(synonyms) - 1

            words[idx] = synonyms[m]

        return " ".join(words)


class Chr_encoder:
    def __init__(self, max_length=1014, alpha_length=70):

        self.alphbets = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'''/\|_@#$%^& *~'+-=<>()[]{}"  # accepted characters in the encoder
        self.alph_to_index = {chr: idx for idx, chr in
                              enumerate(self.alphbets)}  # giving each character an index to use in position later
        self.max_length = max_length  # maximium accepted input length
        self.alpha_length = alpha_length  # accepted character length

    def encode(self, text: str):

        text = text[: self.max_length]
        text = text[::-1]
        text = text.lower()
        encoded_text = torch.zeros((self.alpha_length, self.max_length), dtype=torch.float32)

        for i, chr in enumerate(text):

            if chr in self.alphbets:
                chr_pos = self.alph_to_index[chr]
                encoded_text[chr_pos][i] = 1

        return encoded_text

    def get_alpha(self):
        return self.alph_to_index


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
    splited_train = dataset['train'].train_test_split(train_size=0.8)
    train_d = splited_train['train']
    valid_d = splited_train['test']
    test_d = dataset['test']

    enc = Chr_encoder(max_length=config.text_len, alpha_length=config.alpha_len)
    aug = Augmentor(config.p, config.q)

    if split == "train":
        train_dataset_aug = Chr_dataset(train_d, enc, aug, True)
        train_dataset_unaug = Chr_dataset(train_d, enc, aug, False)
        train_dataset = ConcatDataset([train_dataset_aug, train_dataset_unaug])

        final_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    elif split == "val":
        valid_dataset = Chr_dataset(valid_d, enc, aug, False)
        final_dataloader = DataLoader(valid_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    else:
        test_dataset = Chr_dataset(test_d, enc, aug, False)
        final_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    return final_dataloader
