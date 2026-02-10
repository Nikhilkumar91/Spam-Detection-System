## Spam Detection System 📧🚫

A Machine Learning–based Spam Detection system that classifies messages or emails as Spam or Ham (Not Spam) using text analysis.

----
## Overview

Spam messages are a major issue in emails and messaging platforms.
This project uses Machine Learning and NLP techniques to automatically detect spam with good accuracy and fast predictions.

-----
## Features

Accepts email or message text

Text preprocessing (cleaning & vectorization)

Predicts Spam or Ham (Good Mail)

Shows prediction confidence

Simple and clean web interface

------

## Tech Stack

Python

Machine Learning (NLP)

Scikit-learn

Pandas, NumPy

Flask

HTML, CSS

------
## Dataset

The dataset used for training is taken from Kaggle:

🔗 Dataset Link:
https://www.kaggle.com/datasets/tmehul/spamcsv

The dataset contains labeled messages classified as spam or ham.

----

## How It Works

User enters a message

Text is preprocessed

Features are extracted using vectorization

Trained ML model predicts the class

Result and confidence score are displayed

------
## Project Structure

spam-detection/
```text
├── app.py
├── spam.h5
├── main.py
├── spam.csv
├── requirements.txt
├── templates/
│   └── index.html
├── static/
     └── style.css
```

----

## Output Screenshot

<img width="1904" height="961" alt="image" src="https://github.com/user-attachments/assets/6402ec13-535b-4adf-a8ff-e5c49f705807" />


-----
## Run the Project

git clone <repository-url>

cd spam-detection

pip install -r requirements.txt

python app.py

Open your browser at:
http://127.0.0.1:5000/

----

## Results

Correctly classifies spam and non-spam messages

Fast response time

Suitable for real-time applications

-----

## Future Improvements

Deep Learning models (LSTM, Transformers)

Email inbox integration

Cloud deployment

Multi-language support

-----

## Project Done By:

 V Nikhil Kumar
 
 Aspiring AI/ML Engineer & Data Scientist

 ------

## GitHub:  https://github.com/Nikhilkumar91
