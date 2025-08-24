import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB


@st.cache_data
def load_data():
    data = pd.read_csv('./spam.csv')
    data.drop_duplicates(inplace=True)
    data['Category'] = data['Category'].replace({'ham': 'Not Spam', 'spam': 'Spam'})
    return data

data = load_data()
messages = data['Message']
categories = data['Category']

# Train model 

@st.cache_resource
def train_model(messages, categories):
    x_train, x_test, y_train, y_test = train_test_split(messages, categories, test_size=0.2, random_state=42)
    cv = CountVectorizer(stop_words='english', lowercase=True)
    x_train_features = cv.fit_transform(x_train)
    x_test_features = cv.transform(x_test)
    
    model = MultinomialNB()
    model.fit(x_train_features, y_train)
    accuracy = model.score(x_test_features, y_test)
    
    return model, cv, accuracy

model, cv, accuracy = train_model(messages, categories)

# Streamlit 

st.title("Spam Message Detector")
st.write(f"Model Accuracy: **{accuracy:.2%}**")

user_input = st.text_area("Enter your message here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a message to predict.")
    else:
        features = cv.transform([user_input])
        prediction = model.predict(features)[0]
        if prediction == "Spam":
            st.error("This is a **Spam** message!")
        else:
            st.success("This is **Not Spam**.")
