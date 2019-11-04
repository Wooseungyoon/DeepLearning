# -*- coding: utf-8 -*-

import tensorflow as tf
import pandas as pd
import numpy as np
import os, pickle, time, sys
import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input
from keras.models import Model
from keras import layers
from tensorflow.keras.utils import to_categorical
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

okt = Okt()

class RNN_model():
    def __init__(self):
        self.remove_tag = ['Josa', 'PreEomi', 'Suffix', 'Punctuation', 'Foreign', 'Unknown', 'Hashtag', 'ScreenName', 'Email', 'URL','KoreanParticle']
        self.max_number_char = 70
        self.max_number_word = 30
        
        self.vocab_size_char = 1000
        self.vocab_size_word = 10000
        
        self.embedding_dim_char = 10
        self.embedding_dim_word = 10
        
        self.batch_size_char = 128
        self.batch_size_word = 256
        
        self.epochs = 10
        self.batch_size = 3
        
    def load_data(self):
        pass
    
    def create_intent_embedding(self,data_df):
        tmpData = data_df["intent"]
        intent_index=[]
        for i, intent in enumerate(tmpData):
            intent_index.append(intent)
        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(intent_index)
        intent_box = tokenizer.word_index
        
        seq_intent = tokenizer.texts_to_sequences(intent_index)
        seq_intent_zero = []
        for lst in seq_intent:
            for num in lst:
                seq_intent_zero.append(num-1)
        intent_one_hot = to_categorical(seq_intent_zero)
       
        return np.array(intent_one_hot), len(intent_box)
        # return one_hot coding intent
        
    def create_char_embedding(self, data_df): # return dense vector
        tmpData = data_df["body"]
        char_raw_data = []
        for i, text in enumerate(tmpData):
            r = []
            for j, char in enumerate(text):
                if char !=' ':
                    r.append(char)
            char_raw_data.append(r)
        ## char_raw_data
        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(char_raw_data)
        # create word index
        
        char_seq_data = tokenizer.texts_to_sequences(char_raw_data)
        # change word index into integer index
        
        char_word_index = tokenizer.word_index
        
        char_pad_data = pad_sequences(char_seq_data, 
                                      self.max_number_char, 
                                      padding='pre', value=0.0)
#        print(char_pad_data)
        num_char = min(self.vocab_size_char, len(char_word_index)+1) #vocab_size

        model = Sequential()
        model.add(Embedding(input_dim=num_char,
                            output_dim=self.embedding_dim_char, 
                            input_length=self.max_number_char))
#        model.add(Flatten())
        model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        embedded_matrix = model.predict(char_pad_data)
#       embedded_matrix.reshape(-1,self.max_number_char, self.embedding_dim_char)
        
        return np.array(embedded_matrix)
    
    def create_word2vec(self, data_df, w2v_model_name): #return dense vector
        tmpData = data_df["body"]
        w2v_data =[]
        for i, text in enumerate(tmpData):
            r = []
            tmp = okt.pos(text,norm=True, stem=True)   
            for word in tmp:
                if word not in self.remove_tag:
                    r.append(word[0])
            w2v_data.append(r)
        ## word_raw_data
        
        model = Word2Vec(w2v_data, size=self.embedding_dim_word, window = 4, min_count = 0, sg=1)

        model.init_sims(replace=True)
        model.save(w2v_model_name)
        ## word2vec model 
        
        embedded_vector =[]
        for i, text in enumerate(tmpData):
            r=[]
            tmp = okt.pos(text, norm=True, stem=True)
            for word in tmp:
                if word not in self.remove_tag:
                    r.append(model.wv[word[0]].tolist())
            embedded_vector.append(r)
        
        padded_word = pad_sequences(embedded_vector,
                                    self.max_number_word,
                                    padding='pre', value=0.0)
        
        return np.array(padded_word)
        
    def create_LSTM_model(self,char_embedding, word_embedding, intent_embedding, num_intent):
        ## char_model
        input_char = Input(shape=(self.max_number_char,self.embedding_dim_char ))
        model_char = layers.LSTM(self.batch_size_char,dropout=0.2, recurrent_dropout=0.2)(input_char)
        
        ## word_model
        input_word = Input(shape=(self.max_number_word,self.embedding_dim_word ))
        model_word = layers.LSTM(self.batch_size_word,dropout=0.2, recurrent_dropout=0.2)(input_word)
        
        ## input concatenate
        concatenated = layers.concatenate([model_char,model_word])
        
        ## dense layer
        output = Dense(64, activation='relu')(concatenated)
        output = Dense(num_intent, activation='softmax')(output)
        
        model = Model(inputs=[input_char,input_word],outputs=output)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        print(model.summary())
        
        model.fit([char_embedding,word_embedding],intent_embedding,
                  epochs=self.epochs, 
                  batch_size=self.batch_size,
                  verbose=1,
                  validation_split=0.2,
                  shuffle=True)
        # model learning
        print(model.evaluate([char_embedding,word_embedding],intent_embedding))
        return model
        
    def create_biLSTM_model(self, char_embedding, word_embedding, intent_embedding, num_intent):
        
        input_char = Input(shape=(self.max_number_char,self.embedding_dim_char))
        model_char = layers.Bidirectional(layers.LSTM(self.batch_size_char))(input_char)
        
        input_word = Input(shape=(self.max_number_word,self.embedding_dim_word))
        model_word = layers.Bidirectional(layers.LSTM(self.batch_size_word))(input_word)
        
        concatenated = layers.concatenate([model_char, model_word])
        
        output = layers.Dense(64,activation='relu')(concatenated)
        output = layers.Dense(num_intent,activation='softmax')(output)
        
        model = Model(inputs=[input_char,input_word],outputs=output)
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        print(model.summary())
        
        model.fit([char_embedding, word_embedding], intent_embedding,
                  epochs=self.epochs,
                  batch_size=self.batch_size,
                  verbose=1,
                  validation_split=0.2,
                  shuffle=True)
    
if __name__ == "__main__":
    data_file_path = r'C:\Users\Woo\Desktop\WSY_model'
    os.chdir(data_file_path)
    data_file_name = 'intent_tmp.csv'
    data_df = pd.read_csv(data_file_name,encoding='cp949')[['body', 'intent']]
    # body, intent whole dataframe
    
    w2v_model_name = 'w2v_WSY'
    
    rnn = RNN_model()
    intent, num_intent = rnn.create_intent_embedding(data_df)
    word = rnn.create_word2vec(data_df,w2v_model_name)
    char = rnn.create_char_embedding(data_df)
    rnn.create_LSTM_model(char, word, intent, num_intent)
    rnn.create_biLSTM_model(char, word, intent, num_intent)
    
    