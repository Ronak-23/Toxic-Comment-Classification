from django.shortcuts import render
import numpy as np
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, PorterStemmer
import re
import os
import lightgbm
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer,CountVectorizer
from toxicCommentClassifier.settings import BASE_DIR

W_to_I = {'':0}
add_space_before_punc = lambda x: re.sub(r'(\W|_)', r' \1 ', x)
remove_whitespaces = lambda x: re.sub(r'\s+', ' ', x)
remove_multiples = lambda x: re.sub(r'(.)\1{2,}', r'\1\1', x) #Remove repeated char multiple times

# df_new['clean_text'] = df_new.preprocessed_text.progress_apply(
#     lambda x: remove_whitespaces(remove_multiples(add_space_before_punc(x)))
# )

# # Average len is 44 with min of 1 word and max of 4948.
# df_new['len'] = df_new.clean_text.progress_apply(lambda x: len(x.split()))


def preprocess(text_string):
    stemmer = SnowballStemmer("english")
    try:
        stop_words = set(stopwords.words('english'))
    except:
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
    text_string = text_string.lower() # Convert everything to lower case.
    text_string = re.sub('[^A-Za-z0-9]+', ' ', text_string) # Remove special characters and punctuations
    
    x = text_string.split()
    new_text = []
    
    for word in x:
        if word not in stop_words:
            new_text.append(stemmer.stem(word))
            
    text_string = ' '.join(new_text)
    return text_string

def convert_to_int(word):
    c = W_to_I.get(word, -1)
    if c==-1:
        c = len(W_to_I)
        W_to_I[word] = c
    return c

def convert_text_to_arr(text, max_len=60):
    words = text.split()[:max_len]
    n = len(words)
    if n < max_len:
        words += ['' for _ in range(max_len - n)]
    words = [convert_to_int(word) for word in words]
    return np.array(words)

def main(request):
  context = {}
#   model=pickle.load(open("D:\Projects\Toxic Comment Classification\\toxicCommentClassifier\model\\finalized_model2.sav", 'rb'))
#   X_train = pd.read_csv("D:\\Imp files\\material\\ML labs\\New folder/X_train.csv")
#   y_train = pd.read_csv("D:\\Imp files\\material\\ML labs\\New folder/y_train.csv")
#   print(model)
#   model.fit(X_train, y_train,verbose=20)
  if(request.method == "POST"):
    comment = request.POST['comment']
    # comment_pre = preprocess(comment)
    transformer = TfidfTransformer()
    fp1 = str(os.path.join(BASE_DIR, 'model/feature.pkl'))
    loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open(fp1, "rb")))
    comment_pre = transformer.fit_transform(loaded_vec.fit_transform(np.array([comment])))
    # comment_pre = convert_text_to_arr(comment_pre)
    context["comment"] = comment_pre
    fp2 = str(os.path.join(BASE_DIR, 'model/logreg_model_tfid_vectorizer.sav'))
    model=pickle.load(open(fp2, 'rb'))
    # model = lightgbm.Booster(model_file='D:\Projects\Toxic Comment Classification\\toxicCommentClassifier\model\lgbr_base.txt')
    # Model = model()
    y_pred = model.predict(comment_pre.reshape(1, -1)) 
    # print(model.coef_)
    # y_pred = 0.5
    y_pred=(y_pred>0.50)
    result = "Toxic" if y_pred else "Non Toxic"
    # result = y_pred
    context["result"] = result
  return render(request, 'index.html', context)
