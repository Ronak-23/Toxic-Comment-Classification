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
    context["comment"] = comment_pre
    fp2 = str(os.path.join(BASE_DIR, 'model/logreg_model_tfid_vectorizer.sav'))
    fp3 = str(os.path.join(BASE_DIR, 'model/logreg_model_identity_attack_tfid_vectorizer.sav'))
    fp4 = str(os.path.join(BASE_DIR, 'model/logreg_model_funny_tfid_vectorizer.sav'))
    fp5 = str(os.path.join(BASE_DIR, 'model/logreg_model_insult_tfid_vectorizer.sav'))
    fp6 = str(os.path.join(BASE_DIR, 'model/logreg_model_obscene_tfid_vectorizer.sav'))
    fp7 = str(os.path.join(BASE_DIR, 'model/logreg_model_sexual_explicit_tfid_vectorizer.sav'))
    fp8 = str(os.path.join(BASE_DIR, 'model/logreg_model_threat_tfid_vectorizer.sav'))
    model=pickle.load(open(fp2, 'rb'))
    model1 = pickle.load(open(fp3, 'rb'))
    model2 = pickle.load(open(fp4, 'rb'))
    model3 = pickle.load(open(fp5, 'rb'))
    model4 = pickle.load(open(fp6, 'rb'))
    model5 = pickle.load(open(fp7, 'rb'))
    model6 = pickle.load(open(fp8, 'rb'))

    # model = lightgbm.Booster(model_file='D:\Projects\Toxic Comment Classification\\toxicCommentClassifier\model\lgbr_base.txt')
    # Model = model()
    y_pred = model.predict(comment_pre.reshape(1, -1)) 
    # print(model.coef_)
    # y_pred = 0.5
    y_pred=(y_pred>0.50)
    result = "Toxic" if y_pred else "Non Toxic"
    if(result == "Toxic"):
        y_pred1 = model1.predict(comment_pre.reshape(1, -1))
        y_pred2 = model2.predict(comment_pre.reshape(1, -1))
        y_pred3 = model3.predict(comment_pre.reshape(1, -1))
        y_pred4 = model4.predict(comment_pre.reshape(1, -1))
        y_pred5 = model5.predict(comment_pre.reshape(1, -1))
        y_pred6 = model6.predict(comment_pre.reshape(1, -1))
        types = ""
        if y_pred1:
            types+= "identity attack,"
        if y_pred2:
            types+= "funny,"
        if y_pred3:
            types+= "insult,"
        if y_pred4:
            types+= "obscene,"
        if y_pred5:
            types+= "sexual explicit,"
        if y_pred6:
            types+= "threat,"
        types = types[:-1]
        # result2 = "Funny," if y_pred2 else ""
        # result3 = "Insult," if y_pred3 else ""
        # result4 = "Obscene," if y_pred4 else ""
        # result5 = "Sexual Explicit," if y_pred5 else ""
        # result6 = "Threat" if y_pred6 else ""
        context["types"] = types
        # context["result1"] = result1
        # context["result2"] = result2
        # context["result3"] = result3
        # context["result4"] = result4
        # context["result5"] = result5
        # context["result6"] = result6
    # result = y_pred
    context["result"] = result
  return render(request, 'index.html', context)
