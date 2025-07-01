import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Page setup
st.set_page_config(page_title="AI Stock Predictor", layout="centered")
st.title("📈 AI Stock Price Predictor + GPT-2 Summary")
st.write("🔮 Powered by LSTM (AI) + GPT-2 (Generative AI)")

# Input section
ticker = st.text_input("Enter Stock Ticker Symbol (e.g. AAPL, MSFT, TSLA)", value="AAPL")

if st.button("Predict & Summarize"):
    with st.spinner("Fetching and processing stock data..."):
        # Get stock data
        df = yf.download(ticker, start="2018-01-01", end="2024-12-31")
        df = df[['Close']]
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)

        # Prepare data
        def create_dataset(data, time_step=60):
            x, y = [], []
            for i in range(time_step, len(data)):
                x.append(data[i - time_step:i, 0])
                y.append(data[i, 0])
            return np.array(x), np.array(y)

        x, y = create_dataset(scaled_data)
        x = np.reshape(x, (x.shape[0], x.shape[1], 1))

        # Train model
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(x.shape[1], 1)),
            LSTM(50),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x, y, epochs=5, batch_size=64, verbose=0)

        # Predict future price
        future_input = scaled_data[-60:]
        future_input = np.reshape(future_input, (1, 60, 1))
        pred_scaled = model.predict(future_input)
        predicted_price = scaler.inverse_transform(pred_scaled)[0][0]

    # Plotting
    st.subheader(f"Predicted Next Closing Price: ₹{predicted_price:.2f}")
    fig, ax = plt.subplots()
    ax.plot(df['Close'], label="Historical")
    ax.axhline(predicted_price, color='r', linestyle='--', label="Predicted")
    ax.set_title(f"{ticker} Price Prediction")
    ax.legend()
    st.pyplot(fig)

    # Text generation
    with st.spinner("Generating GPT-2 summary..."):
        prompt = f"Based on stock market analysis, {ticker} is showing potential. Predicted closing price is ₹{predicted_price:.2f}. In summary,"
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model_gpt = GPT2LMHeadModel.from_pretrained("gpt2")
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model_gpt.generate(
            input_ids,
            max_length=100,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.8
        )
        generated_summary = tokenizer.decode(output[0], skip_special_tokens=True)

    st.subheader("🧠 AI-Generated Summary:")
    st.write(generated_summary)
