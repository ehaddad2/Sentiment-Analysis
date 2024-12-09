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

case_normalized = copy.deepcopy(IMDB_dataset)
case_normalized['review'] = case_normalized['review'].apply(normalize_case)
case_normalized.to_csv("case_normalized.csv", index=False)

#2. Punctuation Removal
def remove_punctuation(text):
    text = text.translate(text.maketrans("", "", string.punctuation))
    return text

no_punctuation = copy.deepcopy(IMDB_dataset)
no_punctuation['review'] = no_punctuation['review'].apply(remove_punctuation)
no_punctuation.to_csv("no_punctuation.csv", index=False)


#3. Stop Word Removal
def remove_stop(text):
    stop_words = set(stopwords.words("english"))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    text = " ".join(filtered_words)
    return text

no_stop = copy.deepcopy(IMDB_dataset)
no_stop['review'] = no_stop['review'].apply(remove_stop)
no_stop.to_csv("no_stop.csv", index=False)

#4. Stemming
def stemming(text):
    stemmer = PorterStemmer()
    words = text.split(",")
    stemmed_words = [stemmer.stem(word) for word in words]
    text = ",".join(stemmed_words)
    return text

stemmed = copy.deepcopy(IMDB_dataset)
stemmed['review'] = stemmed['review'].apply(stemming)
stemmed.to_csv("stemmed.csv", index=False)

#5. Lemmatization
def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split(",")
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    text = ",".join(lemmatized_words)
    return text
    
lemmatized = copy.deepcopy(IMDB_dataset)
lemmatized['review'] = lemmatized['review'].apply(lemmatize)
lemmatized.to_csv("lemmatized.csv", index=False)

#6. Removing numbers and symbols
def remove_num_symbols(text):
    text = re.sub(r"[\d#]", "", text)
    return text 

no_num_symbols = copy.deepcopy(IMDB_dataset)
no_num_symbols['review'] = no_num_symbols['review'].apply(remove_num_symbols)
no_num_symbols.to_csv("no_num_symbols.csv", index=False)

#7. removing non-text elements
def remove_non_text(text):
    text = re.sub(r"(<[^>]+>)|(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)", "", text)
    return text

no_non_text = copy.deepcopy(IMDB_dataset)
no_non_text['review'] = no_non_text['review'].apply(remove_non_text)
no_non_text.to_csv("no_non_text.csv", index=False)

#8. Removing extra spaces
def remove_extra_space(text):
    text = re.sub(r"\s+", " ", text)
    return text

no_extra_space = copy.deepcopy(IMDB_dataset)
no_extra_space['review'] = no_extra_space['review'].apply(remove_extra_space)
no_extra_space.to_csv("no_extra_space.csv", index=False)
