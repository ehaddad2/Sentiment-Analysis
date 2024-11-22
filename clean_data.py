import pandas as pd
import numpy as np
import re
import string
import copy
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


IMDB_dataset = pd.read_csv("IMDB Dataset.csv")

#1. Case Normalization
def normalize_case(text):
    text = text.lower()
    return text

#2. Punctuation Removal
def remove_punctuation(text):
    text = text.translate(text.maketrans("", "", string.punctuation))
    return text


#3. Stop Word Removal
def remove_stop(text):
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = " ".join(filtered_words)
    return text

#4. Stemming
def stemming(text):
    stemmer = PorterStemmer()
    words = text.split(",")
    stemmed_words = [stemmer.stem(word) for word in words]
    text = ",".join(stemmed_words)
    return text

#5. Lemmatization
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split(",")
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = ",".join(lemmatized_words)
    return text

#6. Removing numbers and symbols
def remove_num_symbols(text):
    text = re.sub(r"[\d#]", "", text)
    return text 

#7. removing non-text elements
def remove_non_text(text):
    text = re.sub(r"(<[^>]+>)|(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)", "", text)
    return text

#8. Removing extra spaces
def remove_extra_space(text):
    text = re.sub(r"\s+", " ", text)
    return text

def clean_text(text):
    text = normalize_case(text)
    text = remove_punctuation(text)
    text = remove_stop(text)
    text = stemming(text)
    text = lemmatize(text)
    text = remove_num_symbols(text)
    text = remove_non_text(text)
    text = remove_extra_space(text)
    return text

cleaned = copy.deepcopy(IMDB_dataset)
cleaned['review'] = cleaned['review'].apply(clean_text)
cleaned.to_csv("cleaned_IMBD_dataset.csv", index=False)