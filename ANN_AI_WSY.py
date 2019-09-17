# -*- coding: utf-8 -*-

# =============================================================================

import tensorflow as tf
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tqdm import tqdm
import os, pickle, time, sys
from sklearn.preprocessing import OneHotEncoder

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
okt = Okt()
###############################################################################
#     data preprocessing
###############################################################################
def load_data_fn(data_file_name, w2v_model, w2v_size, max_word_num):
    '''
    raw_x : 리뷰글 array나 list
    load_x : 전처리한 리뷰글 데이터
    load_y : 전처리한 리뷰글 긍부정 라벨
    '''
    try:
        data_df = pd.read_csv(data_file_name, encoding = 'utf-8', engine= 'python')
    except UnicodeDecodeError:
        data_df = pd.read_csv(data_file_name, encoding = 'cp949', engine= 'python')
    
#    data_df = data_df.sample(frac=1)
    review_data = data_df[data_df.columns[1]]
    label_data = data_df[data_df.columns[0]]
    raw_x = review_data
    data_len = len(data_df)
    #print(data_len)
    word_vector = w2v_model.wv

#    result = []
#    for i in range(data_len):
#        r = []
#        tmp = okt.pos(review_data.loc[i], norm = True, stem = True)
#        print(tmp[2][1])
#        for word in tmp:
#            if word[1] in ["Noun", "Adjective", "Verb"]:
#                if word[0] in word_vector.vocab:
#                    r.append(word_vector[word[0]])
#        result[i].append(r)
   
    arr_result = np.zeros((data_len,128,300))
    for i in tqdm(range(data_len)):
       tmp = okt.pos(review_data[i], norm=True, stem=True)
       if len(tmp)>128: len_tmp = 128
       else: len_tmp = len(tmp)
       for j in range(len_tmp):
           if tmp[j][1] in ["Noun", "Adjective", "Verb"]:
               if tmp[j][0] in word_vector.vocab:
                   arr_result[i][j] = word_vector[tmp[j][0]]
       
    
#    arr_result = np.zeros((1001,51,301))
#    for i in range(1000):
#        for j in range(len(result[i])):
#            for k in range(len(result[i][j])):
#                arr_result[i][j][k] = result[i][j][k]
    
    load_x = arr_result
    #print(review_data)
    try:
        arr_label = np.array(label_data).reshape(-1,1)
        enc = OneHotEncoder()
        enc.fit(arr_label)
        
        label_onehot = enc.transform(arr_label).toarray()
    #label_onehot
    
        load_y = label_onehot
    except:
        load_y = _
        
    return raw_x, load_x, load_y

###############################################################################
#     create model
###############################################################################
def create_ann_model(tf_model_important_var_name, max_word_num, w2v_size):
    l_1_node = 512
    l_2_node = 256
    n_class = 2
    dropout_rate = 0.2
    learning_rate = 0.001
    
    tf.reset_default_graph()

    x_data = tf.placeholder(tf.float32, [None, max_word_num, w2v_size], name='x_data')   
    re_x_data = tf.reshape(x_data, [-1, max_word_num*w2v_size])
    y_data = tf.placeholder(tf.float32, [None, 2], name='y_data')
    
    
    he_init = tf.contrib.layers.variance_scaling_initializer()
    z_1 = tf.layers.dense(re_x_data, l_1_node, activation=tf.nn.relu, kernel_initializer=he_init)
    
    w_2 = tf.Variable(tf.random_normal([l_1_node, l_2_node], stddev=0.01), name='w_2')
    b_2 = tf.Variable(tf.zeros(shape=(l_2_node)), name='b_2')
    bn_2 = tf.layers.batch_normalization(tf.matmul(z_1, w_2)+b_2, name='bn_2')
    z_2 = tf.nn.relu(bn_2, name='z_2')
    d_2 = tf.nn.dropout(z_2, dropout_rate, name='d_2')
    
    w_3 = tf.Variable(tf.random_normal([l_2_node, 128], stddev=0.01), name='w_3')
    b_3 = tf.Variable(tf.zeros(shape=(128)), name='b_3')
    bn_3 = tf.layers.batch_normalization(tf.matmul(z_2, w_3)+b_3, name='bn_3')
    z_3 = tf.nn.relu(bn_3, name='z_3')
    d_3 = tf.nn.dropout(z_3, dropout_rate, name='d_3')
    
    w_4 = tf.Variable(tf.random_normal([128, n_class], stddev=0.01), name='w_4')
    u_4 = tf.matmul(d_3, w_4, name='u_4')
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=u_4, labels=y_data), name='loss')
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)

    pred_y = tf.nn.softmax(u_4, name='pred_y')
    pred = tf.equal(tf.argmax(pred_y, 1), tf.argmax(y_data, 1), name='pred')
    acc = tf.reduce_mean(tf.cast(pred, tf.float32), name='acc')
    
    for op in [x_data, y_data, loss, acc, pred_y]:
        tf.add_to_collection(tf_model_important_var_name, op)
        
    return x_data, y_data, loss, acc, pred_y, train 
    
###############################################################################
#     early_stopping_and _save_model
###############################################################################
def early_stopping_and_save_model(sess, saver, tf_model_path, save_model_name, input_vali_loss, early_stopping_patience, early_stopping_val_loss_list):
    if len(early_stopping_val_loss_list) != early_stopping_patience:
        early_stopping_val_loss_list = [99.99 for _ in range(early_stopping_patience)]
    
    early_stopping_val_loss_list.append(input_vali_loss)
    if input_vali_loss < min(early_stopping_val_loss_list[:-1]):
        os.chdir(tf_model_path)
        saver.save(sess, './{0}/{0}.ckpt'.format(save_model_name))
        early_stopping_val_loss_list.pop(0)
        
        return True, early_stopping_val_loss_list
    
    elif early_stopping_val_loss_list.pop(0) < min(early_stopping_val_loss_list):
        return False, early_stopping_val_loss_list
    
    else:
        return True, early_stopping_val_loss_list

#################################################################################    
#     model_train
#################################################################################
def model_train(x_data, y_data, loss, acc, pred_y, train, x_train, y_train, x_vali, y_vali, batch_size, epoch_num, tf_model_path, tf_model_name, early_stopping_patience):
    batch_index_list = list(range(0, x_train.shape[0], batch_size))
    vali_batch_index_list = list(range(0, x_vali.shape[0], batch_size))
    
    train_loss_list, vali_loss_list = [], []
    train_acc_list, vali_acc_list = [], []
    
    start_time = time.time()
    saver = tf.train.Saver()
    early_stopping_val_loss_list = []
    print('\n%s\n%s - training....'%('-'*100, tf_model_name))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
                
        for epoch in range(epoch_num):
            total_loss, total_acc, vali_total_loss, vali_total_acc = 0, 0, 0, 0
            processing_bar_var = [0, 0]
            
            train_random_seed = int(np.random.random()*10**4)
            for x in [x_train, y_train]:
                np.random.seed(train_random_seed)
                np.random.shuffle(x)
                
            for i in batch_index_list:
                batch_x, batch_y = x_train[i:i+batch_size], y_train[i:i+batch_size]
                
                processing_bar_var[0] += len(batch_x)
                processing_bar_print = int(processing_bar_var[0]*100/len(x_train))-processing_bar_var[1]
                if processing_bar_print != 0:
                    sys.stdout.write('-'*processing_bar_print)
                    sys.stdout.flush()
    
                processing_bar_var[1] += (int(processing_bar_var[0]*100/len(x_train))-processing_bar_var[1])
                
                _, loss_val, acc_val = sess.run([train, loss, acc], feed_dict={x_data: batch_x, y_data: batch_y})
                total_loss += loss_val
                total_acc += acc_val
                
            train_loss_list.append(total_loss/len(batch_index_list))
            train_acc_list.append(total_acc/len(batch_index_list))
            
            sys.stdout.write('\n#%4d/%d%s' % (epoch + 1, epoch_num, '  |  '))
            sys.stdout.write('Train_loss={:.4f} / Train_acc={:.4f}{}'.format(train_loss_list[-1], train_acc_list[-1], '  |  '))
            sys.stdout.flush()
                    
            for i in vali_batch_index_list:
                vali_batch_x, vali_batch_y = x_vali[i:i+batch_size], y_vali[i:i+batch_size]
                
                vali_loss_val, vali_acc_val = sess.run([loss, acc], feed_dict={x_data: vali_batch_x, y_data: vali_batch_y})
                vali_total_loss += vali_loss_val
                vali_total_acc += vali_acc_val
                    
            vali_loss_list.append(vali_total_loss/len(vali_batch_index_list))
            vali_acc_list.append(vali_total_acc/len(vali_batch_index_list))
            
            tmp_running_time = time.time() - start_time
            sys.stdout.write('Vali_loss={:.4f} / Vali_acc={:.4f}{}'.format(vali_loss_list[-1], vali_acc_list[-1], '  |  '))
            sys.stdout.write('%dm %5.2fs\n'%(tmp_running_time//60, tmp_running_time%60))
            sys.stdout.flush()
            
            bool_continue, early_stopping_val_loss_list = early_stopping_and_save_model(sess, saver, tf_model_path, tf_model_name, vali_loss_list[-1], early_stopping_patience, early_stopping_val_loss_list)
            if not bool_continue:
                print('{0}\nstop epoch : {1}\n{0}'.format('-'*100, epoch-early_stopping_patience+1))
                break
            
    
    running_time = time.time() - start_time
    print('%s\ntraining time : %d m  %5.2f s\n%s'%('*'*100, running_time//60, running_time%60, '*'*100))
    
    
    os.chdir(r'{}\{}'.format(tf_model_path, tf_model_name))
    epoch_list = [i for i in range(1, epoch+2)]
    graph_loss_list = [train_loss_list, vali_loss_list, 'r', 'b', 'loss', 'upper right', '{}_loss.png'.format(tf_model_name)]
    graph_acc_list = [train_acc_list, vali_acc_list, 'r--', 'b--', 'acc', 'lower right', '{}_acc.png'.format(tf_model_name)]
    for train_l_a_list, vali_l_a_list, trian_color, vali_color, loss_acc, legend_loc, save_png_name in [graph_loss_list, graph_acc_list]:
        plt.plot(epoch_list, train_l_a_list, trian_color, label='train_'+loss_acc)
        plt.plot(epoch_list, vali_l_a_list, vali_color, label='validation_'+loss_acc)
        plt.xlabel('epoch')
        plt.ylabel(loss_acc)
        plt.legend(loc=legend_loc)
        plt.title(tf_model_name)
        plt.savefig(save_png_name)
        plt.show()

###############################################################################
#     model test
###############################################################################
def model_test(x_test, y_test, tf_model_path, tf_model_name, tf_model_important_var_name):
    tf.reset_default_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        os.chdir(tf_model_path)
        saver = tf.train.import_meta_graph('./{0}/{0}.ckpt.meta'.format(tf_model_name))
        saver.restore(sess, './{0}/{0}.ckpt'.format(tf_model_name))
        x_data, y_data, loss, acc, pred_y = tf.get_collection(tf_model_important_var_name)
        
        test_loss, test_acc, test_pred_y, test_true_y = sess.run([loss, acc, pred_y, y_data], feed_dict={x_data: x_test, y_data: y_test})
        
    f_pred_y = np.argmax(test_pred_y, axis=1)
    f_true_y = np.argmax(test_true_y, axis=1)
    print('\n' + '='*100)
    print(classification_report(f_true_y, f_pred_y, target_names=['Positive', 'Negative']))
    print(pd.crosstab(pd.Series(f_true_y), pd.Series(f_true_y), rownames=['True'], colnames=['Predicted'], margins=True))
    print('\n{}\nTest_loss = {:.4f}\nTest_acc = {:.4f}\n{}'.format('='*100, test_loss, test_acc, '='*100))
        
###############################################################################
#     predict label
###############################################################################
def predict_label(tf_model_path, tf_model_name, tf_model_important_var_name, raw__model_test, x_model_test, pred_label_file_path, pred_label_file_name):
    tf.reset_default_graph()
    os.chdir(tf_model_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.import_meta_graph('./{0}/{0}.ckpt.meta'.format(tf_model_name))
        saver.restore(sess, './{0}/{0}.ckpt'.format(tf_model_name))
        x_data, y_data, loss, acc, pred_y = tf.get_collection(tf_model_important_var_name)
        
        hw_y_pred = sess.run(pred_y, feed_dict={x_data: x_model_test})
        f_pred_y = np.argmax(hw_y_pred, axis=1)
        
    os.chdir(pred_label_file_path)
    pd.DataFrame({'label':f_pred_y, 'review':raw_model_test}).to_csv(pred_label_file_name, index=False)
        
###############################################################################
###############################################################################
    
if __name__ == "__main__":
# =============================================================================
    data_file_path = r'C:\Users\Woo\Desktop\AI'
    train_file_name = 'train_data_WSY.csv'
    validation_file_name = 'validation_data_WSY.csv'
    test_file_name = 'test_data_WSY.csv'
    
    w2v_file_path = r'C:\Users\Woo\Desktop\AI'
    os.chdir(w2v_file_path)
    w2v_model = Word2Vec.load('word2vec_WSY.model')
    
#    save_model_path = r'C:\Users\Woo\Desktop\BIZ랩\tf_model'
#    save_model_name = 'hw_5_ANN_model_WSY'
    tf_model_path = r'C:\Users\Woo\Desktop\AI\tf_modell'
    tf_model_name = 'ANN_model_WSY_def'
    tf_model_important_var_name = 'important_vars_ops'
    
    model_test_data_file_path = r'C:\Users\Woo\Desktop\AI'
    model_test_data_no_label_file_name = 'model_test_data_no_label.csv'
    
    pred_label_file_path = r'C:\Users\Woo\Desktop\AI'
    pred_label_file_name = 'ann_predict_label_WSY.csv'
# =============================================================================  
    w2v_size = 300
    max_word_num = 128
    
    early_stopping_patience = 10
    epoch_num = 100
    batch_size = 512
# =============================================================================   

    os.chdir(data_file_path)
    raw_train, x_train, y_train = load_data_fn(train_file_name, w2v_model, w2v_size, max_word_num)
    raw_vali, x_vali, y_vali = load_data_fn(validation_file_name, w2v_model, w2v_size, max_word_num)
    raw_test, x_test, y_test = load_data_fn(test_file_name, w2v_model, w2v_size, max_word_num)
    
    os.chdir(model_test_data_file_path)
    raw_model_test, x_model_test, _ = load_data_fn(model_test_data_no_label_file_name, w2v_model, w2v_size, max_word_num)
    
    print('train_shape : {} / {}'.format(x_train.shape, y_train.shape))
    print('validation_shape : {} / {}'.format(x_vali.shape, y_vali.shape))
    print('test_shape : {} / {}'.format(x_test.shape, y_test.shape))
    print('model_test_shape : {}'.format(x_model_test.shape))
    
    x_data, y_data, loss, acc, pred_y, train = create_ann_model(tf_model_important_var_name, max_word_num, w2v_size)

    model_train(x_data, y_data, loss, acc, pred_y, train, x_train, y_train, x_vali, y_vali, batch_size, epoch_num, tf_model_path, tf_model_name, early_stopping_patience)
    model_test(x_test, y_test, tf_model_path, tf_model_name, tf_model_important_var_name)
    predict_label(tf_model_path, tf_model_name, tf_model_important_var_name, raw_model_test, x_model_test, pred_label_file_path, pred_label_file_name)
    