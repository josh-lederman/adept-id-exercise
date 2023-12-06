#!pip install sentence_transformers
import pandas as pd
import numpy as np
from sklearn import linear_model, metrics
from sentence_transformers import SentenceTransformer
import pickle

# Load of trained and saved logistic regression classifier
with open('classifier.model', 'rb') as handle:
  classifier = pickle.load(handle)

def predict_n_onets(titles, n):
  # Arguments:
  # titles - Pandas Series-like of title text
  # n - Integer, representing number of top ONET matches to return
  title_embedding = SentenceTransformer(
      'sentence-transformers/all-MiniLM-L6-v2').encode(titles)
  onet_probs = classifier.predict_proba(title_embedding)
  top_n_ind = [np.argsort(-prob)[0:n] for prob in onet_probs]
  return np.array([classifier.classes_[i] for i in top_n_ind])
