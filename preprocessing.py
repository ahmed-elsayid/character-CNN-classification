import torch
import numpy as np
import random
import nltk
from nltk.corpus import wordnet


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
