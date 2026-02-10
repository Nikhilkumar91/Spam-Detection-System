import numpy as np
import pandas as pd
import sklearn
import nltk
import os
import sys
import logging
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from log_code import setup_logging
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense,Activation,Flatten,LSTM,Bidirectional,SimpleRNN,Embedding,Masking,GRU
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.activations import sigmoid,relu,softmax,tanh
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')
logger=setup_logging('main')
nltk.download('stopwords',quiet=True)
nltk.download('wordnet',quiet=True)

class NLP:
  try:
      def __init__(self):
          self.vocab_size = 5000
  except Exception as e:
      er_ty, er_msg, er_tb = sys.exc_info()
      line_number = er_tb.tb_lineno
      logger.info("Error Type:", er_ty)
      logger.info("Error Message:", er_msg)
      logger.info("Error Line No:", line_number)


  def load_data(self):
      try:
          self.df=pd.read_csv('C:\\Users\\nikhi\\Downloads\\Data science practice\\Spam Detection\\spam.csv',encoding='latin-1')
          self.df=self.df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],axis=1)
          #self.df['SeniorCitizen']=self.df['SeniorCitizen'].map({0:'No',1:'Yes'})
          self.df=self.df.rename(columns={'v1':'Target','v2':'Mails'})
          self.df['Target']=self.df['Target'].map({'spam':0,'ham':1}).astype(int)
          logger.info(f"Sample Data\n: {self.df.sample(7)}")
          logger.info(f"Checking Null values \n: {self.df.isnull().sum()}")
          logger.info(f"Info about tha Data \n: {self.df.describe()}")

      except Exception as e:
          er_ty, er_msg, er_tb = sys.exc_info()
          line_number = er_tb.tb_lineno
          logger.info("Error Type:", er_ty)
          logger.info("Error Message:", er_msg)
          logger.info("Error Line No:", line_number)

  def preprocess(self):
      try:
          cleaned_data=[]
          for i in self.df['Mails']:
              text=i
              text=''.join([i for i in text if i not in string.punctuation])
              text=text.lower()
              lemm = WordNetLemmatizer()
              text=' '.join([lemm.lemmatize(i) for i in text.split() if i not in stopwords.words('english')])
              cleaned_data.append(text)
              #self.df['Cleaned_Mails']=cleaned_data

          logger.info(f"Original data: {self.df.head(10)}")
          #print(cleaned_data[:10])
          print(len(cleaned_data))
          print(len(self.df))

          assert len(cleaned_data) == len(self.df)
          self.df['Cleaned_Mails']=cleaned_data


          logger.info(f"Cleaned data\n: {self.df.head()}")

      except Exception as e:
          er_ty, er_msg, er_tb = sys.exc_info()
          line_number = er_tb.tb_lineno
          logger.info("Error Type:", er_ty)
          logger.info("Error Message:", er_msg)
          logger.info("Error Line No:", line_number)

  def balance(self):
      try:
          spam = self.df[self.df['Target'] == 0]
          ham = self.df[self.df['Target'] == 1]

          if len(spam) > len(ham):
              # oversample ham
              ham_balanced = resample(
                  ham,
                  replace=True,
                  n_samples=len(spam),
                  random_state=42
              )
              spam_balanced = spam
          else:
              # oversample spam
              spam_balanced = resample(
                  spam,
                  replace=True,
                  n_samples=len(ham),
                  random_state=42
              )
              ham_balanced = ham

          self.df_balanced = pd.concat([spam_balanced, ham_balanced]).sample(
              frac=1, random_state=42
          )

          logger.info("Balanced class counts:")
          logger.info(self.df_balanced['Target'].value_counts())
          logger.info(self.df_balanced.head(10))
          logger.info(self.df_balanced.info())
          # Removing Original Mails column
          self.df_balanced=self.df_balanced.drop(['Mails'],axis=1)
          #logger.info(f"Data After Removing the original Mail column:\n{self.df_balanced.info()}")
          #print(self.df_balanced['Target'].value_counts())

      except Exception as e:
          er_ty, er_msg, er_tb = sys.exc_info()
          line_number = er_tb.tb_lineno
          logger.info("Error Type:", er_ty)
          logger.info("Error Message:", er_msg)
          logger.info("Error Line No:", line_number)

  def conversion(self):
      try:
          self.dic_size = 2000
          self.vectors = [one_hot(i, self.dic_size) for i in self.df_balanced['Cleaned_Mails']]
          print(self.vectors[:10])

          #finding maximum length of the mail so that we decrease the memory size
          self.t = []
          for i in self.vectors:
              self.t.append(len(i))
          print(max(self.t))

          self.input_vectors = pad_sequences(self.vectors, maxlen=max(self.t), padding='post')
          #print(self.input_vectors[:10])
          print(self.df_balanced['Target'].shape)


      except Exception as e:
          er_ty, er_msg, er_tb = sys.exc_info()
          line_number = er_tb.tb_lineno
          logger.info("Error Type:", er_ty)
          logger.info("Error Message:", er_msg)
          logger.info("Error Line No:", line_number)

  def training(self):
      try:

          self.y = np.array(self.df_balanced['Target'])  # binary labels
          self.model = Sequential()

          self.model.add(Embedding(
              input_dim=self.dic_size,
              output_dim=50,
              input_length=max(self.t),
              mask_zero=True
          ))

          self.model.add(Bidirectional(LSTM(32, return_sequences=True)))
          self.model.add(Bidirectional(LSTM(16)))

          self.model.add(Dense(1, activation='sigmoid'))

          self.model.compile(
              optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy']
          )

          self.model.summary()
          '''self.model.fit(
              self.input_vectors,
              self.y,
              epochs=25,
              batch_size=50
          )'''

          # =========================
          # SAVE MODEL
          # =========================
          self.model.save('spam_model.h5')

          # =========================
          # LOAD MODEL
          # =========================
          #self.m = load_model('spam_model.h5')




      except Exception as e:
          er_ty, er_msg, er_tb = sys.exc_info()
          line_number = er_tb.tb_lineno
          logger.info("Error Type:", er_ty)
          logger.info("Error Message:", er_msg)
          logger.info("Error Line No:", line_number)

  def predict_mail(self, mail):
      try:
          labels = ['spam', 'ham']
          lemm=WordNetLemmatizer()

          text = mail.lower()
          text = ''.join([i for i in text if i not in string.punctuation])
          text = ' '.join(
              lemm.lemmatize(w)
              for w in text.split()
              if w not in stopwords.words('english')
          )

          v = [one_hot(text, self.dic_size)]
          p = pad_sequences(v, maxlen=80, padding='post')

          prob = self.m.predict(p)[0][0]
          print("Probability:", prob)
          return "ham" if prob >= 0.4 else "Spam"

      except Exception as e:
          er_ty, er_msg, er_tb = sys.exc_info()
          line_number = er_tb.tb_lineno
          logger.info("Error Type:", er_ty)
          logger.info("Error Message:", er_msg)
          logger.info("Error Line No:", line_number)





if __name__=='__main__':
    try:
        obj=NLP()
        obj.load_data()
        obj.preprocess()
        obj.balance()
        obj.conversion()
        obj.training()
        obj.m = load_model('spam_model.h5')
        #print("Probability:", prob)
        print(obj.predict_mail("The product is very good I recommend other to use it"))



    except Exception as e:
        er_ty, er_msg, er_tb = sys.exc_info()
        line_number = er_tb.tb_lineno
        logger.info("Error Type:", er_ty)
        logger.info("Error Message:", er_msg)
        logger.info("Error Line No:", line_number)
