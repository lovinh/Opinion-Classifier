import math
import numpy as np
import pandas as pd
import os
import string
import re

def features_extraction(corpus: list[list[str]]) -> set[str]:
    features: set = set()
    for document in corpus:
        for term in document:
            features.add(term)
    return features

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
    '''
    Lớp cho phép tạo một từ điển các từ mang đặc trưng của một lớp cụ thể nào đó trích xuất từ một tập dữ liệu.
    
    Đặc trưng ở đây tức là ứng với từ này thì mang hàm ý thuộc về lớp này nhiều hơn. Nếu trong câu gặp từ đặc trưng của một lớp nào đó thì ta có cơ sở để kết luận câu đó có vẻ thuộc lớp mà từ đó đặc trưng.
    
    Parameters
    ---
        metrics: str. Chỉ có hai giá trị là ```gini``` và ```entropy```. Phương thức đo độ tạp chất của từ. Nếu ```metrics='gini'```, độ tạp chất được đo bởi điểm số gini. Ngược lại đo bởi điểm số Entropy.
 
    '''
    def __init__(self, metrics: str) -> None:
        self.__metrics = metrics

    def fit(self, corpus: list[list[str]] | np.ndarray, labels_corpus: list[str] | np.ndarray | pd.Series) -> None:
        '''
        Áp dụng kho văn bản và tập nhãn tương ứng với kho văn bản để khớp vào mô hình.
        
        Mô hình sẽ tính toán phân phối tần số của tập từ phân biệt trích ra từ kho văn bản theo các lớp trong tập nhãn. Đồng thời tính toán độ tạp chất tương ứng của các từ trong tập từ phân biệt đó trong các lớp.

        Parameters
        ---
            corpus: Kho văn bản. Là một ma trận (n,). Mỗi hàng biểu diễn một văn bản. Mỗi văn bản là một mảng (1,) gồm nhiều từ đơn vị.

            labels_corpus: Tập nhãn ứng với từng văn bản. Kích thước (n,1)
        
        Return
        ---
            None
        '''
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
        '''
        Tạo ra tập từ điển đặc trưng dựa trên các tham số chỉ định đặc biệt. Các dữ liệu tính toán phải được tính sau bước fit.
        
        Parameters
        ---
            threshold: Một giá trị thực biểu thị mốc tạp chất tối đa để từ đó được coi là đặc trưng. Nếu độ tạp chất của từ vượt qua mốc, từ đó không được coi là đặc trưng của lớp nào.

            min_samples: Một giá trị nguyên biểu thị tổng số văn bản mà từ đó thuộc vào tối thiểu để được coi là đủ đặc trưng. Nếu số văn bản chứa từ đó ít hơn min_samples, tức là từ đó không đủ phổ biến trong kho văn bản để được coi là đặc trưng.
        
            is_sorted: Giá trị boolean. Nếu is_sorted=```True```, tập từ điển được sắp xếp theo chiều tăng trong bảng ascii và ngược lại.

            is_reversed: Giá trị boolean biểu thị sắp xếp theo chiều tăng dần hay giảm dần. Bị bỏ qua nếu is_sorted=```False```.

        Return
        ---
            Một dictionary chứa các từ đặc trưng cho từng lớp của kho văn bản.

        '''
        self.__feature_vocabulary: dict[str, str] = {}
        score_vocabulary = self.get_impute_score(is_sorted, is_reversed)
        for key in score_vocabulary:
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

    def fit_transform(self, corpus: list[list[str]] | np.ndarray, labels_corpus: list[str] | np.ndarray | pd.Series, threshold: float = 0.0, min_samples: int = 0, is_sorted: bool = True, is_reversed: bool = False) -> dict[str, str]:
        '''
        Áp dụng kho văn bản và tập nhãn tương ứng với kho văn bản để khớp vào mô hình và sau đó tạo ra tập từ điển đặc trưng dựa trên các tham số chỉ định đặc biệt. 
        
        Mô hình sẽ tính toán phân phối tần số của tập từ phân biệt trích ra từ kho văn bản theo các lớp trong tập nhãn. Đồng thời tính toán độ tạp chất tương ứng của các từ trong tập từ phân biệt đó trong các lớp.

        Sau khi tính toán phân phối, dựa theo tham số chỉ định tạo ra tập từ điển đặc trưng. 
        
        Parameters
        ---
            corpus: Kho văn bản. Là một ma trận (n,). Mỗi hàng biểu diễn một văn bản. Mỗi văn bản là một mảng (1,) gồm nhiều từ đơn vị.

            labels_corpus: Tập nhãn ứng với từng văn bản. Kích thước (n,1)
        
            threshold: Một giá trị thực biểu thị mốc tạp chất tối đa để từ đó được coi là đặc trưng. Nếu độ tạp chất của từ vượt qua mốc, từ đó không được coi là đặc trưng của lớp nào.

            min_samples: Một giá trị nguyên biểu thị tổng số văn bản mà từ đó thuộc vào tối thiểu để được coi là đủ đặc trưng. Nếu số văn bản chứa từ đó ít hơn min_samples, tức là từ đó không đủ phổ biến trong kho văn bản để được coi là đặc trưng.
        
            is_sorted: Giá trị boolean. Nếu is_sorted=```True```, tập từ điển được sắp xếp theo chiều tăng trong bảng ascii và ngược lại.

            is_reversed: Giá trị boolean biểu thị sắp xếp theo chiều tăng dần hay giảm dần. Bị bỏ qua nếu is_sorted=```False```.

        Return
        ---
            Một dictionary chứa các từ đặc trưng cho từng lớp của kho văn bản.
        '''
        self.fit(corpus, labels_corpus)
        return self.transform(threshold, min_samples, is_sorted, is_reversed)

    @property
    def labels(self) -> list[str] | np.ndarray | pd.Series:
        '''
        Trả về tập nhãn của kho văn bản
        '''
        return self.__labels

    @property
    def frequency_distributions(self) -> dict[str, dict[str, int]]:
        '''
        Trả về bảng tần suất của từng lớp.
        '''
        return self.__frequency_distributions

    def get_frequency_distributions(self, class_label: str) -> dict[str, int]:
        """
        cho phép truy xuất bảng tần suất của một lớp cụ thể
        """
        if not (class_label in self.__labels):
            raise KeyError(class_label)
        return self.__frequency_distributions.get(class_label)

    @property
    def impute_score(self) -> dict[str, float]:
        """
        Trả về độ tạp chất của từng từ phân biệt của kho văn bản.
        """
        return self.__score_vocabulary

    def get_impute_score(self, is_sorted: bool = True, is_reversed: bool = False) -> dict[str, float]:
        """
        Trả về độ tạp chất của từng từ phân biệt của kho văn bản. Tuy nhiên cho phép sắp xếp
        """
        if is_sorted:
            return dict(sorted(self.__score_vocabulary.items(), reverse=is_reversed, key=lambda x: x[1]))
        return self.__score_vocabulary

    @property
    def feature_vocabulary(self) -> dict[str, str]:
        """
        Trả về danh sách từ điển đặc trưng
        """
        return self.__feature_vocabulary

    def get_feature_vocabulary(self, labels : str) -> list[str]:
        res = []
        for key in self.__feature_vocabulary:
            if self.__feature_vocabulary[key] == labels:
                res.append(key)
        return res
