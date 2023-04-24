import math


def tf(document: list[str], term: str) -> float:
    tf_score = 0
    for words in document:
        if term == words:
            tf_score += 1
    return tf_score


def idf(corpus: list[list[str]], term: str) -> float:
    document_contain_term: int = 0
    for document in corpus:
        if term in document:
            document_contain_term += 1
    idf_score = float((1 + len(corpus)) / (document_contain_term + 1))
    return math.log(idf_score) + 1


def tf_idf(corpus: list[list[str]], document: list[str], term: str) -> float:
    return tf(document, term) * idf(corpus, term)


def features_extraction(corpus: list[list[str]]) -> set[str]:
    features: set = set()
    for document in corpus:
        for term in document:
            features.add(term)
    return features


def count_vectorizer(corpus: list[list[str]], vocabulary: list[str], fq: bool = True) -> list[list[int]]:
    vector: list[list[int]] = []
    for sen in corpus:
        sen_count: list[int] = []
        for term in vocabulary:
            sen_count.append(tf(sen, term) if fq else (
                1 if term in sen else 0))
        vector.append(sen_count)
    return vector


def tf_idf_vectorizer(corpus: list[list[str]], vocabulary: list[str]) -> list[dict[str, float]]:
    tf_idf_dict: list[dict[str, float]] = []

    for sen in corpus:
        tf_idf_sen: dict[str, float] = {}
        for term in sen:
            tf_idf_sen[term] = tf_idf(corpus, sen, term)
        tf_idf_dict.append(tf_idf_sen)

    tf_idf_corpus: list[list[float]] = []

    for sentence in tf_idf_dict:
        tf_idf_sentence: list[float] = []
        for term in vocabulary:
            tf_idf_sentence.append(sentence.get(term, 0))
        tf_idf_corpus.append(tf_idf_sentence)

    return tf_idf_corpus
