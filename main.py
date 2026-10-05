import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index

word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pretrained model

model = load_model('simple_rnn_imdb.keras')

# Function to decode reviews

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Prediction function

def predict_sentiment(review):
    preprocessed_input = preprocess_text(review)

    prediction = model.predict(preprocessed_input)

    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    return sentiment, prediction[0][0]

# Streamlit app

import streamlit as st


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sentify | IMDB Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎬 Sentify")

st.subheader(
    "AI-Powered Movie Review Sentiment Analysis"
)

st.write(
    "Enter a movie review below and let the trained "
    "SimpleRNN model determine whether it is positive "
    "or negative."
)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.info(
    "🧠 **Model:** SimpleRNN  |  "
    "🔤 **Vocabulary:** 10,000 words  |  "
    "📏 **Sequence Length:** 500"
)


# ============================================================
# USER INPUT
# ============================================================

st.header("📝 Enter your movie review")

user_input = st.text_area(
    "Movie Review",
    height=180,
    placeholder=(
        "Example: This movie was absolutely fantastic! "
        "The acting was brilliant and the story kept "
        "me engaged..."
    )
)


# ============================================================
# EXAMPLE REVIEWS
# ============================================================

st.write("💡 **Try an example:**")

col1, col2 = st.columns(2)


with col1:

    positive_example = st.button(
        "😊 Positive Example",
        use_container_width=True
    )


with col2:

    negative_example = st.button(
        "😡 Negative Example",
        use_container_width=True
    )


# ============================================================
# EXAMPLE BUTTON LOGIC
# ============================================================

if positive_example:

    user_input = (
        "This movie was absolutely amazing. "
        "The story was beautiful and the acting "
        "was excellent. I really enjoyed every "
        "minute of it."
    )

    st.text_area(
        "Selected Review",
        value=user_input,
        height=120
    )


if negative_example:

    user_input = (
        "This movie was terrible and extremely "
        "boring. The story was weak and the acting "
        "was disappointing. I would not recommend "
        "watching it."
    )

    st.text_area(
        "Selected Review",
        value=user_input,
        height=120
    )


# ============================================================
# ANALYZE SENTIMENT
# ============================================================

st.write("")


if st.button(
    "🔮 Analyze Sentiment",
    use_container_width=True
):

    if not user_input.strip():

        st.warning(
            "⚠️ Please enter a movie review first."
        )

    else:

        # Run the existing prediction function
        sentiment, prediction = predict_sentiment(
            user_input
        )

        prediction = float(prediction)


        # ====================================================
        # RESULT
        # ====================================================

        if sentiment == "Positive":

            confidence = prediction

            st.success(
                f"😊 **Positive Review**\n\n"
                f"The model is "
                f"**{confidence * 100:.2f}%** confident "
                f"that this review is positive."
            )

        else:

            confidence = 1 - prediction

            st.error(
                f"😞 **Negative Review**\n\n"
                f"The model is "
                f"**{confidence * 100:.2f}%** confident "
                f"that this review is negative."
            )


        # ====================================================
        # PROBABILITY
        # ====================================================

        st.subheader("📊 Prediction Probability")

        st.progress(
            prediction
        )


        col1, col2 = st.columns(2)


        with col1:

            st.metric(
                "😊 Positive",
                f"{prediction * 100:.2f}%"
            )


        with col2:

            st.metric(
                "😞 Negative",
                f"{(1 - prediction) * 100:.2f}%"
            )


# ============================================================
# ABOUT THE MODEL
# ============================================================

with st.expander("🧠 About the Model"):

    st.write(
        """
        ### SimpleRNN Architecture

        Movie Review  
        ↓  
        Word Encoding  
        ↓  
        Padding (500 tokens)  
        ↓  
        Embedding Layer  
        ↓  
        SimpleRNN (128 units)  
        ↓  
        Dense + Sigmoid  
        ↓  
        Positive / Negative
        """
    )

    st.write("### Model Configuration")

    st.write(
        """
        - **Dataset:** IMDB Movie Reviews
        - **Vocabulary Size:** 10,000
        - **Maximum Sequence Length:** 500
        - **Embedding Dimension:** 128
        - **RNN Units:** 128
        - **Output:** Binary Classification
        - **Optimizer:** Adam
        - **Loss:** Binary Crossentropy
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Built with Python • TensorFlow • Keras • Streamlit"
)