from flask import Flask, render_template, request, jsonify
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

app = Flask(__name__)

# Load your models and vectorizers here
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

ps = PorterStemmer()
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    text = [i for i in text if i.isalnum()]
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
    text = [ps.stem(i) for i in text]
    return " ".join(text)

@app.route('/')
def index():
    return render_template('sms.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.form.get('sms')
    transformed_sms = transform_text(input_text)
    vector_input = tfidf.transform([transformed_sms])  # Note the list here
    result = model.predict(vector_input)[0]
    msg = 'Spam' if result == 1 else 'Not Spam'
    return jsonify({'msg': msg})

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask,render_template,request,jsonify
# import streamlit as st
# import pickle
# from nltk.corpus import stopwords
# import string
# from nltk.stem.porter import PorterStemmer
# ps = PorterStemmer()
# import nltk
# nltk.download('stopwords')
# from sklearn.naive_bayes import MultinomialNB


# app = Flask(__name__)

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb')) 

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)
#     text = [i for i in text if i.isalnum()]
#     text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]
#     text = [ps.stem(i) for i in text]
#     return " ".join(text)

# @app.route('/')
# def index():
#     return render_template('sms.html')


# @app.route('/predict',methods=['POST'])
# def predict():
#     msg = None
#     # input = request.form.get("sms")

#     transformed_sms = transform_text(input)
#     vector_input = tfidf.transform([transformed_sms])
#     result = model.predict(vector_input)[0]
#     if result==1:
#         msg = 'Spam'
#     else:
#         msg='Not Spam'
#     return jsonify({'msg':msg})


# if __name__ == '__main__':
#     app.run(debug=True)