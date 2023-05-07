
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import json
from libs.tf_idf import *
from libs.utils import *
import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import pickle
sys.path.append(
    r"E:\Learn Machine Learning\Project\Opinion Classifier\res\tf-idf\libs")


def tokenization(text):
    return text.split()


def tf_idf_transform(idf, vocabulary, sentence):

    res = {}
    for term in vocabulary:
        res[term] = 0

    for token in sentence:
        if token in vocabulary:
            res[token] = tf(sentence, token) * idf.get(token, 0)

    return pd.DataFrame([res])


if __name__ == "__main__":
    model = load_model(
        r"E:\Learn Machine Learning\Project\Opinion Classifier\res\tf-idf\logistic_regression.pkl")

    with open(r"E:\Learn Machine Learning\Project\Opinion Classifier\res\tf-idf\idf_term.pkl", "rb") as f:
        idf_term: dict[str, float] = pickle.load(f)

    data: pd.DataFrame = read_dataset(
        r"E:\Learn Machine Learning\Project\Opinion Classifier\res\dataset\tf-idf-not-remv-stopwords.csv", "csv")
    vocabulary = data.columns.to_list()
    data = data.drop(columns=data.columns.values[0], axis=1)
    data = data.drop(columns=data.columns.values[5774], axis=1)

    vocabulary = data.columns.to_list()

    while True:

        sentence = input("Nhập câu: ('\q' to exit): ")

        if sentence == '\q':
            break

        tokens = tokenization(sentence)

        X = tf_idf_transform(idf_term, vocabulary, tokens)

        # print(X.columns)

        prediction = model.predict([X.iloc[0]])
        if prediction == [0]:
            print("Dự đoán: Positive")
        if prediction == [2]:
            print("Dự đoán: Negative")
        if prediction == [1]:
            print("Dự đoán: Neutral")
        print('Dự đoán: ' + str(model.predict([X.iloc[0]])))
        print("Độ chính xác dự đoán: " + str(max(model.predict_proba([X.iloc[0]]))))
