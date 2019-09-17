# -*- coding: utf-8 -*-

# 
# =============================================================================
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
from tqdm import tqdm
from pprint import pprint
import pandas as pd
import os
import numpy as np

okt = Okt()

def fn_data_split(data_df, train_file_name, vali_file_name, test_file_name):
    #data_df0x = data_df[data_df.label==0].index
    #data_df0 = data_df.loc[data_df["label"]==0]
    
    #data_idx0 = data_df[data_df.label == 0].index
    #data_idx1x = data_df[data_df.label == 1].index
    cnt_data0 = len(data_df[data_df["label"] ==0])
    data_idx0 = data_df.loc[data_df.label == 0][:]
    data_idx1x = data_df.loc[data_df.label == 1][:]
    data_idx1 = data_idx1x.sample(n=cnt_data0, random_state=1)
    #df.loc[:,'A']
    #data_idx1 = np.random.choice(data_idx1x,cnt_data0, replace=False)
    
    #data_idx0
    #sample = np.concatenate([data_df0, data_df1])
    #df = data_df.loc[sample]
    #data_df1 = data_df.loc[data_df["label"]==1]
    #print(data_df0x.head())
    #print(data_df0.head())
    #print(df.count())
    #print(data_idx0[3])
    train0,temp0 = train_test_split(data_idx0,test_size=0.4,random_state=42)
    vali0,test0 = train_test_split(temp0,test_size=0.5,random_state=42)
    train1,temp1 = train_test_split(data_idx1,test_size=0.4,random_state=42)
    vali1,test1 = train_test_split(temp1,test_size=0.5,random_state=42)
    
    
    train = np.concatenate([train0,train1])
    vali = np.concatenate([vali0,vali1])
    test = np.concatenate([test0,test1])
    
    train_df = pd.DataFrame(train)
    vali_df = pd.DataFrame(vali)
    test_df = pd.DataFrame(test)
    
    train_df.to_csv(train_file_name,header=False,index=False)
    vali_df.to_csv(vali_file_name,header=False,index=False)
    test_df.to_csv(test_file_name,header=False,index=False)
#    print(len(train_df))
#    print(len(vali_df))
#    print(len(test_df))
##    train_df = data_df.loc[train]
#    vali_df = data_df.loc[vali]
#    test_df = data_df.loc[test]
    #train_df = pd.DataFrame(train)
    
    
    #print(train_file_name.head())
    #train1,temp1 = train_test_split(data_df1,test_size=0.4,random_state=42)
    #vali1,test1 = train_test_split(temp1,test_size=0.5,random_state=42)
    
def fn_create_word2vec(data_df, w2v_model_name):
    tmpData = data_df["review"]
    data_len = len(data_df)
    result = []
    for i in range(data_len):
        r = []
        tmp = okt.pos(tmpData.loc[i], norm = True, stem = True)
        for word in tmp:
            if word[1] in ["Noun", "Adjective", "Verb"]:
                r.append(word[0])
        result.append(r)
    
    model = Word2Vec(result, size=300, window = 5, min_count = 10, sg=1)
    model.init_sims(replace=True)
    model.save(w2v_model_name)
    #print(result)
    #print(data_len)
#    data = [okt.nouns(tmpData.loc[i]) for i in range(data_len)]
#    print(data.head())    
        
    return model




if __name__ == "__main__":
    train_file_name = 'train_data_WSY.csv'
    vali_file_name = 'validation_data_WSY.csv'
    test_file_name = 'test_data_WSY.csv'
    
    w2v_model_name = 'hw_4_word2vec_WSY.model'
    
    os.chdir(r'C:\Users\Woo\Desktop\AI')
    data_file_name = 'data.csv'
    data_df = pd.read_csv(data_file_name, encoding='cp949')[['label', 'review']]
    print('data_df shape - ', data_df.shape)
    
    data_df = data_df.drop_duplicates()
    print('data_df(drop_duplicates) shape - ', data_df.shape)
    
    print(data_df.groupby(['label'])['label'].count())
    data_df
# =============================================================================

    os.chdir(r'C:\Users\Woo\Desktop\AI')
    fn_data_split(data_df, train_file_name, vali_file_name, test_file_name)
    w2v_model = fn_create_word2vec(data_df, w2v_model_name)
#    w2v_model = Word2Vec.load(w2v_model_name)
    print(w2v_model.most_similar("의자",topn=5))
    print(w2v_model.wv.similarity('검', '배송'))
    for test_word in ['월요일' , '배송', '빠르다', '좋다', '감사', '별로']:
        print('*'*50 + '\n' + test_word)
        pprint(w2v_model.wv.most_similar(test_word, topn=5))

            

'''

**************************************************
월요일
[('금요일', 0.959784746170044),
 ('목요일', 0.956427276134491),
 ('수요일', 0.9455667734146118),
 ('화요일', 0.9429277777671814),
 ('토욜', 0.9383894801139832)]
**************************************************
배송
[('파른', 0.7278835773468018),
 ('리오네', 0.7141883373260498),
 ('송도', 0.7126674652099609),
 ('명절', 0.6764638423919678),
 ('송이', 0.6738250255584717)]
**************************************************
빠르다
[('감솨', 0.7879383563995361),
 ('파른', 0.7672903537750244),
 ('총알', 0.7666589021682739),
 ('빨랏', 0.7610298991203308),
 ('빨르다', 0.7538026571273804)]
**************************************************
좋다
[('잘삿어', 0.7105856537818909),
 ('정말로', 0.7042036056518555),
 ('욧', 0.7030841708183289),
 ('아영', 0.7001270055770874),
 ('좋아욤', 0.6972665786743164)]
**************************************************
감사
[('하비다', 0.7892617583274841),
 ('감솨', 0.7831730842590332),
 ('고맙다', 0.7772567272186279),
 ('감사하다', 0.7344866394996643),
 ('힙니', 0.7170889973640442)]
**************************************************
별로
[('별루', 0.6891950368881226),
 ('그닥', 0.6276479959487915),
 ('역다', 0.5523307919502258),
 ('이외', 0.5468701124191284),
 ('안좋다', 0.5464729070663452)]

'''
