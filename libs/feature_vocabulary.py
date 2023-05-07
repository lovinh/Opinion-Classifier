import math
import numpy as np
import pandas as pd
from tf_idf import features_extraction
import os
from utils import *
import string
import re

string_pattele = string.punctuation + '\n'


def remove_emoji(string):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def remove_punctuation(text, except_char: str = None):
    try:
        result_text = "".join(
            char for char in text if char not in string_pattele or (except_char and char == except_char))
        result_text = remove_emoji(result_text)
    except TypeError:
        result_text = None
    return result_text


def filter_none_text_id(documents: pd.Series | list[str]) -> list[int]:
    need_filter: list[str] = [text for text in documents]
    none_text_index: list[int] = []
    for i, text in enumerate(need_filter):
        if (text is None):
            print(str(i) + ": " + str(text))
            none_text_index.append(i)

    return none_text_index


def gini_score(n_class_samples: list[int]) -> float:
    gini: float = 0
    n_samples: int = 0
    for n in n_class_samples:
        n_samples += n
    for n in n_class_samples:
        gini += (n / n_samples) ** 2

    return 1 - gini


def entropy_score(n_class_samples: list[int]) -> float:
    entropy: float = 0
    n_samples = 0
    for n in n_class_samples:
        n_samples += n

    for n in n_class_samples:
        entropy += (n / n_samples) * math.log(n / n_samples)

    return -entropy


class feature_vocabulary_transfer:
    def __init__(self, metrics: str) -> None:
        self.__metrics = metrics

    def fit(self, corpus: list[list[str]] | np.ndarray, labels_corpus: list[str] | np.ndarray | pd.Series) -> None:
        self.__labels_corpus: list[str] | np.ndarray | pd.Series = labels_corpus
        self.__labels: list[str] = list(set(labels_corpus))
        self.__corpus: list[list[str]] | np.ndarray | pd.DataFrame = corpus
        self.__frequency_distributions: dict[str, dict[str, int]] = {
            label: {} for label in self.__labels
        }

        corpus_vocabulary = list(features_extraction(corpus))
        corpus_vocabulary.sort()

        self.__corpus_vocabulary: np.ndarray = np.array(
            corpus_vocabulary)
        print(self.__corpus_vocabulary)
        for distribution_label in self.__frequency_distributions:
            for token in self.__corpus_vocabulary:
                self.__frequency_distributions[distribution_label][token] = 0

            for sentence, label in zip(self.__corpus, self.__labels_corpus):
                if label == distribution_label:
                    for term in sentence:
                        self.__frequency_distributions[distribution_label][term] += 1

            self.__frequency_distributions[distribution_label] = dict(sorted(
                self.__frequency_distributions[distribution_label].items(), key=lambda x: x[1], reverse=True))

        self.__score_vocabulary: dict[str, float] = {}
        if self.__metrics == "gini":
            for token in self.__corpus_vocabulary:
                self.__score_vocabulary[token] = gini_score(
                    [self.__frequency_distributions[x][token]
                        for x in self.__frequency_distributions]
                )
        else:
            for token in self.__corpus_vocabulary:
                self.__score_vocabulary[token] = entropy_score(
                    [self.__frequency_distributions[x][token]
                        for x in self.__frequency_distributions]
                )

    def transform(self, threshold: float = 0.0, min_samples: int = 0, is_sorted: bool = True, is_reversed: bool = False) -> dict[str, str]:
        self.__feature_vocabulary: dict[str, str] = {}
        score_vocabulary = self.get_impute_score(is_sorted, is_reversed)
        print(score_vocabulary)
        for key in score_vocabulary:
            # print(key)
            total_sample = 0
            for distribution in self.__frequency_distributions:
                total_sample += self.__frequency_distributions[distribution][key]

            if score_vocabulary[key] <= threshold and total_sample >= min_samples:
                freq_max = 0
                for distribution in self.__frequency_distributions:
                    freq_max = max(
                        freq_max, self.__frequency_distributions[distribution][key])

                for distribution in self.__frequency_distributions:
                    if freq_max == self.__frequency_distributions[distribution][key]:
                        self.__feature_vocabulary[key] = distribution

        return self.__feature_vocabulary

    @property
    def labels(self) -> list[str] | np.ndarray | pd.Series:
        return self.__labels

    @property
    def frequency_distributions(self) -> dict[str, dict[str, int]]:
        return self.__frequency_distributions

    def get_frequency_distributions(self, class_label: str) -> dict[str, int]:
        if not (class_label in self.__labels):
            raise KeyError(class_label)
        return self.__frequency_distributions.get(class_label)

    @property
    def impute_score(self) -> dict[str, float]:
        return self.__score_vocabulary

    def get_impute_score(self, is_sorted: bool = True, is_reversed: bool = False) -> dict[str, float]:
        if is_sorted:
            return dict(sorted(self.__score_vocabulary.items(), reverse=is_reversed, key=lambda x: x[1]))
        return self.__score_vocabulary

    @property
    def feature_vocabulary(self) -> dict[str, str]:
        return self.__feature_vocabulary
