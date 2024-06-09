import streamlit as st
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from PIL import Image

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Define the preprocess_text function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)  # Remove punctuation
    text = re.sub('\d+', '', text)  # Remove numbers
    text = text.split()  # Tokenize text
    text = [word for word in text if word not in stopwords.words('english')]  # Remove stopwords
    return ' '.join(text)

# Streamlit app
st.title('Sentiment Analysis of Hotel Reviews')

# Load the model and vectorizer
with open('sentiment_model.pkl', 'rb') as f:
    model, vectorizer = pickle.load(f)

# Load images
positive_image = Image.open(r'C:\Users\Rahul\Desktop\IPL\t20_cricket\env\Hotels_India\poisitive_image.jpeg')
neutral_image = Image.open(r'C:\Users\Rahul\Desktop\IPL\t20_cricket\env\Hotels_India\neutral_image.jpg')
negative_image = Image.open(r'C:\Users\Rahul\Desktop\IPL\t20_cricket\env\Hotels_India\negative_image.jpg')

# User input
review = st.text_area('Enter a hotel review:')
if st.button('Analyze'):
    review = preprocess_text(review)
    review_vec = vectorizer.transform([review])
    sentiment = model.predict(review_vec)[0]
    st.write(f'Sentiment: {sentiment}')
    
    # Display corresponding image
    if sentiment == 'positive':
        st.image(positive_image, caption='Positive Sentiment', use_column_width=True)
    elif sentiment == 'neutral':
        st.image(neutral_image, caption='Neutral Sentiment', use_column_width=True)
    else:
        st.image(negative_image, caption='Negative Sentiment', use_column_width=True)
