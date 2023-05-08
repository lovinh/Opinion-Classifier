import string
import re
import warnings
import pandas as pd
string_pattele = string.punctuation + '\n'
string_pattele


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
        warnings.warn(f"Có lỗi trong quá trình loại bỏ ký tự đặc biệt của '{text}'. Giá trị trả về None.")
        result_text = None
    return result_text

def filter_none_text_id(documents : pd.Series | list[str]) -> list[int]:
    need_filter : list[str] = [text for text in documents]
    none_text_index : list[int] = []
    for i, text in enumerate(need_filter):
        if (text is None):
            print(str(i) + ": " + str(text))
            none_text_index.append(i)
    
    return 

def lower_case(text):
    if text != None:
        return text.lower()
    
def tokenization(text):
    word_list = text.split()
    token = list()
    for word in word_list:
        if word != "":
            token.append(word)
    return token