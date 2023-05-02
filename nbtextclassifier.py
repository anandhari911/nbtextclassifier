import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
import pickle
btext = pd.read_csv("bbc-text.txt")
btext = btext.rename(columns={'text': 'News_Headline'}, inplace=False)
btext.category = btext.category.map({'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4})
vector = CountVectorizer(stop_words='english', lowercase=False)
X = btext.News_Headline
y = btext.category
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6, random_state=1)
vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_test_transformed = vector.transform(X_test)
nb = MultinomialNB()
nb.fit(X_transformed, y_train)
with open('model.pkl', 'wb') as f:
    pickle.dump(nb, f)
st.header("NB Text Classification")
input=st.text_area("Please enter the text",value="")
if st.button("Predict"):
    input_text=vector.transform([input].toarray())
    predict=nb.predict(input_text)[0]
    predict_map={0: 'TECH', 1: 'BUSINESS', 2: 'SPORTS', 3: 'ENTERTAINMENT', 4: 'POLITICS'}
    result=predict_map[predict]
    st.write(f"Predicted category: {result}")

    

