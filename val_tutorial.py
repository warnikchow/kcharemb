####### Read files and Find class weights

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data
    
fci = read_data('data/fci_train_val.txt')
fci_data = [t[1] for t in fci]
fci_label= [int(t[0]) for t in fci]

fci_test = read_data('data/fci_test.txt')
fci_test_data = [t[1] for t in fci_test]
fci_test_label= [int(t[0]) for t in fci_test]

from sklearn.utils import class_weight
class_weights_fci = class_weight.compute_class_weight('balanced', np.unique(fci_label), fci_label)

####### Import dictionaries

import numpy as np
import fasttext
model_ft = fasttext.load_model('vectors/model_drama.bin')

from numpy import linalg as la

kor_char = np.load('kor_char.npy').item()

######## Character-level embedding function

import hgtk
import han2one
from han2one import shin_onehot, cho_onehot, char2onehot
alp = han2one.alp 
uniquealp = han2one.uniquealp 

def featurize_rnnchar(corpus,wdim,chardict,maxlen):
    rnn_shin  = np.zeros((len(corpus),maxlen*3,len(alp)))
    rnn_cho   = np.zeros((len(corpus),maxlen*3,len(alp)+len(uniquealp)))
    rnn_char  = np.zeros((len(corpus),maxlen,len(alp)))
    rnn_onehot= np.zeros((len(corpus),maxlen,len(chardict)))
    rnn_total = np.zeros((len(corpus),maxlen,wdim))
    for i in range(len(corpus)):
        if i%1000 ==0:
            print(i)
        s = corpus[i]
        for j in range(len(s)):
            if j < maxlen and hgtk.checker.is_hangul(s[-j-1])==True:
                if j>0:
                    rnn_shin[i][-3*j-3:-3*j,:] = np.transpose(shin_onehot(s[-j-1]))
                    rnn_cho[i][-3*j-3:-3*j,:] = np.transpose(cho_onehot(s[-j-1]))
                else:
                    rnn_shin[i][-3*j-3:,:] = np.transpose(shin_onehot(s[-j-1]))
                    rnn_cho[i][-3*j-3:,:] = np.transpose(cho_onehot(s[-j-1]))
                rnn_char[i][-j-1,:] = char2onehot(s[-j-1])
                if s[-j-1] in model_ft:
                    rnn_total[i][-j-1,:] = model_ft[s[-j-1]]
                if s[-j-1] in chardict:
                    rnn_onehot[i][-j-1,chardict[s[-j-1]]]=1
    return rnn_shin, rnn_cho, rnn_onehot, rnn_total, rnn_char

fci_rec_shin, fci_rec_cho, fci_rec_onehot, fci_rec, fci_rec_char = featurize_rnnchar(fci_data,100,kor_char,80)

####### Import Keras and Define F1-score function

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))
from keras.models import Sequential
import keras.layers as layers
from keras import optimizers
adam_half = optimizers.Adam(lr=0.0005)
from keras.callbacks import ModelCheckpoint

from keras.callbacks import Callback
from sklearn import metrics
class Metricsf1macro(Callback):
    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []
        self.val_f1s_w = []
        self.val_recalls_w = []
        self.val_precisions_w = []
    def on_epoch_end(self, epoch, logs={}):
        val_predict = np.asarray(self.model.predict(self.validation_data[0]))
        val_predict = np.argmax(val_predict,axis=1)
        val_targ = self.validation_data[1]
        _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
        _val_f1_w = metrics.f1_score(val_targ, val_predict, average="weighted")
        _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
        _val_recall_w = metrics.recall_score(val_targ, val_predict, average="weighted")
        _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
        _val_precision_w = metrics.precision_score(val_targ, val_predict, average="weighted")
        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)
        self.val_f1s_w.append(_val_f1_w)
        self.val_recalls_w.append(_val_recall_w)
        self.val_precisions_w.append(_val_precision_w)
        print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
        print("— val_f1_w: %f — val_precision_w: %f — val_recall_w: %f"%(_val_f1_w, _val_precision_w, _val_recall_w))

metricsf1macro = Metricsf1macro()

class Metricsf1macro_forself(Callback):
 def on_train_begin(self, logs={}):
  self.val_f1s = []
  self.val_recalls = []
  self.val_precisions = []
  self.val_f1s_w = []
  self.val_recalls_w = []
  self.val_precisions_w = []
 def on_epoch_end(self, epoch, logs={}):
  if len(self.validation_data)>2:
   val_predict = np.asarray(self.model.predict([self.validation_data[0],self.validation_data[1]]))
   val_predict = np.argmax(val_predict,axis=1)
   val_targ = self.validation_data[2]
  else:
   val_predict = np.asarray(self.model.predict(self.validation_data[0]))
   val_predict = np.argmax(val_predict,axis=1)
   val_targ = self.validation_data[1]
  _val_f1 = metrics.f1_score(val_targ, val_predict, average="macro")
  _val_f1_w = metrics.f1_score(val_targ, val_predict, average="weighted")
  _val_recall = metrics.recall_score(val_targ, val_predict, average="macro")
  _val_recall_w = metrics.recall_score(val_targ, val_predict, average="weighted")
  _val_precision = metrics.precision_score(val_targ, val_predict, average="macro")
  _val_precision_w = metrics.precision_score(val_targ, val_predict, average="weighted")
  self.val_f1s.append(_val_f1)
  self.val_recalls.append(_val_recall)
  self.val_precisions.append(_val_precision)
  self.val_f1s_w.append(_val_f1_w)
  self.val_recalls_w.append(_val_recall_w)
  self.val_precisions_w.append(_val_precision_w)
  print("— val_f1: %f — val_precision: %f — val_recall: %f"%(_val_f1, _val_precision, _val_recall))
  print("— val_f1_w: %f — val_precision_w: %f — val_recall_w: %f"%(_val_f1_w, _val_precision_w, _val_recall_w))

metricsf1macro_self = Metricsf1macro_forself()

####### Construct BiLSTM & BiLSTM-SA models

from keras.layers import LSTM
from keras.layers import Bidirectional

def validate_bilstm(result,y,hidden_lstm,hidden_dim,cw,val_sp,bat_size,filename):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_lstm), input_shape=(len(result[0]), len(result[0][0]))))
    model.add(layers.Dense(hidden_dim, activation='relu'))
    model.add(layers.Dense(int(max(y)+1), activation='softmax'))
    model.summary()
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro,checkpoint]
    model.fit(result,y,validation_split=val_sp,epochs=50,batch_size=bat_size,callbacks=callbacks_list,class_weight=cw)

validate_bilstm(fci_rec_shin,fci_label,32,128,class_weights_fci,0.1,16,'model/tutorial/rec_shin')
validate_bilstm(fci_rec_cho,fci_label,32,128,class_weights_fci,0.1,16,'model/tutorial/rec_cho')
validate_bilstm(fci_rec_onehot,fci_label,32,128,class_weights_fci,0.1,16,'model/tutorial/rec_char_onehot')
validate_bilstm(fci_rec,fci_label,32,128,class_weights_fci,0.1,16,'model/tutorial/rec')
validate_bilstm(fci_rec_char,fci_label,32,128,class_weights_fci,0.1,16,'model/tutorial/rec_char')

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Lambda, TimeDistributed
import keras.backend as K
from keras.layers.core import Dropout

def validate_rnn_self_drop(x_rnn,x_y,hidden_lstm,hidden_con,hidden_dim,cw,val_sp,bat_size,filename):
    char_r_input = Input(shape=(len(x_rnn[0]),len(x_rnn[0][0])),dtype='float32')
    r_seq = Bidirectional(LSTM(hidden_lstm,return_sequences=True))(char_r_input)
    r_att = Dense(hidden_con, activation='tanh')(r_seq)
    att_source   = np.zeros((len(x_rnn),hidden_con))
    att_test     = np.zeros((len(x_rnn),hidden_con))
    att_input    = Input(shape=(hidden_con,), dtype='float32')
    att_vec      = Dense(hidden_con,activation='relu')(att_input)
    att_vec      = Dropout(0.3)(att_vec)
    att_vec      = Dense(hidden_con,activation='relu')(att_vec)
    att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([att_vec,r_att])
    att_vec = Dense(len(x_rnn[0]),activation='softmax')(att_vec)
    att_vec = layers.Reshape((len(x_rnn[0]),1))(att_vec)
    r_seq   = layers.multiply([att_vec,r_seq])
    r_seq   = Lambda(lambda x: K.sum(x, axis=1))(r_seq)
    r_seq   = Dense(hidden_dim, activation='relu')(r_seq)
    r_seq   = Dropout(0.3)(r_seq)
    r_seq   = Dense(hidden_dim, activation='relu')(r_seq)
    r_seq   = Dropout(0.3)(r_seq)
    main_output = Dense(int(max(x_y)+1),activation='softmax')(r_seq)
    model = Sequential()
    model = Model(inputs=[char_r_input,att_input],outputs=[main_output])
    model.summary()
    model.compile(optimizer=adam_half,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro_self,checkpoint]
    model.fit([x_rnn,att_source],x_y,validation_split=val_sp,epochs=50,batch_size= bat_size ,callbacks=callbacks_list,class_weight=cw)

validate_rnn_self_drop(fci_rec_shin,fci_label,32,64,256,class_weights_fci,0.1,16,'model/tutorial/rec_self_shin')
validate_rnn_self_drop(fci_rec_cho,fci_label,32,64,256,class_weights_fci,0.1,16,'model/tutorial/rec_self_cho')
validate_rnn_self_drop(fci_rec_onehot,fci_label,32,64,256,class_weights_fci,0.1,16,'model/tutorial/rec_self_onehot')
validate_rnn_self_drop(fci_rec,fci_label,32,64,256,class_weights_fci,0.1,16,'model/tutorial/rec_self')
validate_rnn_self_drop(fci_rec_char,fci_label,32,64,256,class_weights_fci,0.1,16,'model/tutorial/rec_self_char')

####### Test phase

from keras.models import load_model

fci_model_shin = load_model('model/tutorial/rec_shin-45-0.8789-0.7717.hdf5')
fci_model_cho  = load_model('model/tutorial/rec_cho-44-0.8734-0.7569.hdf5')
fci_model_onehot = load_model('model/tutorial/rec_char_onehot-13-0.8772-0.7867.hdf5')
fci_model      = load_model('model/tutorial/rec-19-0.8868-0.7955.hdf5')
fci_model_char = load_model('model/tutorial/rec_char-43-0.8798-0.7850.hdf5')

fci_self_model_shin = load_model('model/tutorial/rec_self_shin-45-0.8828-0.7781.hdf5')
fci_self_model_cho  = load_model('model/tutorial/rec_self_cho-50-0.8819-0.7903.hdf5')
fci_self_model_onehot = load_model('model/tutorial/rec_self_onehot-07-0.8767-0.7891.hdf5')
fci_self_model      = load_model('model/tutorial/rec_self-45-0.8896-0.8110.hdf5')
fci_self_model_char = load_model('model/tutorial/rec_self_char-49-0.8825-0.7934.hdf5')

fci_test_shin, fci_test_cho, fci_test_onehot, fci_test, fci_test_char = featurize_rnnchar(fci_test_data, 100, kor_char, 80)

def test_bilstm(model, corpus, answer):
    if len(corpus)>10000:
        disp = 10000
    else:
        disp = 1000
    pred = np.zeros(len(corpus))
    for i in range(len(corpus)):
        rec = corpus[i].reshape(1,len(corpus[0]),len(corpus[0][0]))
        if i%disp ==0:
            print(i)
        z = model.predict(rec)[0]
        z = np.argmax(z)
        pred[i] = int(z)
    acc = metrics.accuracy_score(answer, pred)
    f1 = metrics.f1_score(answer, pred, average="macro")
    print('Accuracy: ', acc,' F1: ', f1)

test_bilstm(fci_model_shin, fci_test_shin, fci_test_label)
test_bilstm(fci_model_cho, fci_test_cho, fci_test_label)
test_bilstm(fci_model_onehot, fci_test_onehot, fci_test_label)
test_bilstm(fci_model, fci_test, fci_test_label)
test_bilstm(fci_model_char, fci_test_char, fci_test_label)

def test_bilstm_self(model, hidden_con, corpus, answer):
    if len(corpus)>10000:
        disp = 10000
    else:
        disp = 1000
    pred = np.zeros(len(corpus))
    for i in range(len(corpus)):
        rec = corpus[i].reshape(1,len(corpus[0]),len(corpus[0][0]))
        att=np.zeros((1,hidden_con))
        if i%disp ==0:
            print(i)
        z = model.predict([rec,att])[0]
        z = np.argmax(z)
        pred[i]=int(z)
    acc = metrics.accuracy_score(answer, pred)
    f1 = metrics.f1_score(answer, pred, average="macro")
    print('Accuracy: ', acc,' F1: ', f1)

test_bilstm_self(fci_self_model_shin, 64, fci_test_shin, fci_test_label)
test_bilstm_self(fci_self_model_cho, 64, fci_test_cho, fci_test_label)
test_bilstm_self(fci_self_model_onehot, 64, fci_test_onehot, fci_test_label)
test_bilstm_self(fci_self_model, 64, fci_test, fci_test_label)
test_bilstm_self(fci_self_model_char, 64, fci_test_char, fci_test_label)

####### Repeat for NSMC

def read_data_nsmc(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        data = data[1:] # header ƃŗ
    return data

nsmc_train = read_data_nsmc('data/ratings_train.txt')
nsmc_test = read_data_nsmc('data/ratings_test.txt')

nsmc_data = [t[1] for t in nsmc_train]
nsmc_label= [int(t[2]) for t in nsmc_train]

nsmc_test_data = [t[1] for t in nsmc_test]
nsmc_test_label= [int(t[2]) for t in nsmc_test]

from sklearn.utils import class_weight
class_weights_nsmc = class_weight.compute_class_weight('balanced', np.unique(nsmc_label), nsmc_label)

####### Featurization w/o 1hot

def featurize_rnnchar_no1hot(corpus,wdim,chardict,maxlen):
    rnn_shin  = np.zeros((len(corpus),maxlen*3,len(alp)))
    rnn_cho   = np.zeros((len(corpus),maxlen*3,len(alp)+len(uniquealp)))
    rnn_char  = np.zeros((len(corpus),maxlen,len(alp)))
    rnn_total = np.zeros((len(corpus),maxlen,wdim))
    for i in range(len(corpus)):
        if i%1000 ==0:
            print(i)
        s = corpus[i]
        for j in range(len(s)):
            if j < maxlen and hgtk.checker.is_hangul(s[-j-1])==True:
                if j>0:
                    rnn_shin[i][-3*j-3:-3*j,:] = np.transpose(shin_onehot(s[-j-1]))
                    rnn_cho[i][-3*j-3:-3*j,:] = np.transpose(cho_onehot(s[-j-1]))
                else:
                    rnn_shin[i][-3*j-3:,:] = np.transpose(shin_onehot(s[-j-1]))
                    rnn_cho[i][-3*j-3:,:] = np.transpose(cho_onehot(s[-j-1]))
                rnn_char[i][-j-1,:] = char2onehot(s[-j-1])
                if s[-j-1] in model_ft:
                    rnn_total[i][-j-1,:] = model_ft[s[-j-1]]
    return rnn_shin, rnn_cho, rnn_total, rnn_char

nsmc_rec_shin, nsmc_rec_cho, nsmc_rec, nsmc_rec_char = featurize_rnnchar_no1hot(nsmc_data,100,kor_char,140)

####### Validation w/o 1hot

validate_bilstm(nsmc_rec_shin,nsmc_label,32,128,class_weights_nsmc,0.1,64,'model/tutorial_nsmc/rec_shin')
validate_bilstm(nsmc_rec_cho,nsmc_label,32,128,class_weights_nsmc,0.1,64,'model/tutorial_nsmc/rec_cho')
validate_bilstm(nsmc_rec,nsmc_label,32,128,class_weights_nsmc,0.1,64,'model/tutorial_nsmc/rec')
validate_bilstm(nsmc_rec_char,nsmc_label,32,128,class_weights_nsmc,0.1,64,'model/tutorial_nsmc/rec_char')

validate_rnn_self_drop(nsmc_rec_shin,nsmc_label,32,64,256,class_weights_nsmc,0.1,64,'model/tutorial_nsmc/rec_self_shin')
validate_rnn_self_drop(nsmc_rec_cho,nsmc_label,32,64,256,class_weights_nsmc,0.1,64,'model/tutorial_nsmc/rec_self_cho')
validate_rnn_self_drop(nsmc_rec,nsmc_label,32,64,256,class_weights_nsmc,0.1,64,'model/tutorial_nsmc/rec_self_char_dense')
validate_rnn_self_drop(nsmc_rec_char,nsmc_label,32,64,256,class_weights_nsmc,0.1,64,'model/tutorial_nsmc/rec_self_char')

####### Featurization & Validation for 1hot (due to memory capacity)

def featurize_onehotonly(corpus,chardict,maxlen):
    rnn_onehot= np.zeros((len(corpus),maxlen,len(chardict)))
    for i in range(len(corpus)):
        s = corpus[i]
        for j in range(len(s)):
            if j < maxlen and hgtk.checker.is_hangul(s[-j-1])==True:
                if s[-j-1] in chardict:
                    rnn_onehot[i][-j-1,chardict[s[-j-1]]]=1
    return rnn_onehot

nsmc_tr_temp = nsmc_train[:135000]
nsmc_val = nsmc_train[135000:]
nsmc_val_data = [t[1] for t in nsmc_val]
nsmc_val_label = [int(t[2]) for t in nsmc_val]

def validate_bilstm_onehot(y,hidden_lstm,hidden_dim,cw,bat_size,filename):
    model = Sequential()
    model.add(Bidirectional(LSTM(hidden_lstm), input_shape=(140, 2534)))
    model.add(layers.Dense(hidden_dim, activation='relu'))
    model.add(layers.Dense(int(max(y)+1), activation='softmax'))
    model.summary()
    model.compile(optimizer=adam_half, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro,checkpoint]
    for i in range(50):
        print(i,'th epoch / 50')
        shuffle(nsmc_tr_temp)
        ###### shuffled!
        nsmc_tr_shuffled = [t[1] for t in nsmc_tr_temp]
        nsmc_tr_label= [int(t[2]) for t in nsmc_tr_temp]
        onehot = featurize_onehotonly(nsmc_tr_shuffled[:120000],kor_char,140)
        onehot_y = nsmc_tr_label[:120000]
        model.fit(onehot,onehot_y,epochs=1,batch_size=bat_size,class_weight=cw)
        onehot = featurize_onehotonly(nsmc_tr_shuffled[120000:]+nsmc_val_data,kor_char,140)
        onehot_y = nsmc_tr_label[120000:]+nsmc_val_label
        model.fit(onehot,onehot_y,validation_split=0.5,epochs=1,batch_size=bat_size,callbacks=callbacks_list,class_weight=cw)

validate_bilstm_onehot(nsmc_label,32,128,class_weights_nsmc,64,'model/tutorial_nsmc/rec_onehot')

def validate_rnn_self_drop_onehot(x_y,hidden_lstm,hidden_con,hidden_dim,cw,bat_size,filename):
    char_r_input = Input(shape=(140,2534),dtype='float32')
    r_seq = Bidirectional(LSTM(hidden_lstm,return_sequences=True))(char_r_input)
    r_att = Dense(hidden_con, activation='tanh')(r_seq)
    att_input    = Input(shape=(hidden_con,), dtype='float32')
    att_vec      = Dense(hidden_con,activation='relu')(att_input)
    att_vec      = Dropout(0.3)(att_vec)
    att_vec      = Dense(hidden_con,activation='relu')(att_vec)
    att_vec = Lambda(lambda x: K.batch_dot(*x, axes=(1,2)))([att_vec,r_att])
    att_vec = Dense(140,activation='softmax')(att_vec)
    att_vec = layers.Reshape((140,1))(att_vec)
    r_seq   = layers.multiply([att_vec,r_seq])
    r_seq   = Lambda(lambda x: K.sum(x, axis=1))(r_seq)
    r_seq   = Dense(hidden_dim, activation='relu')(r_seq)
    r_seq   = Dropout(0.3)(r_seq)
    r_seq   = Dense(hidden_dim, activation='relu')(r_seq)
    r_seq   = Dropout(0.3)(r_seq)
    main_output = Dense(int(max(x_y)+1),activation='softmax')(r_seq)
    model = Sequential()
    model = Model(inputs=[char_r_input,att_input],outputs=[main_output])
    model.summary()
    model.compile(optimizer=adam_half,loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    filepath=filename+"-{epoch:02d}-{val_acc:.4f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, mode='max')
    callbacks_list = [metricsf1macro_self,checkpoint]
    for i in range(50):
        print(i,'th epoch / 50')
        shuffle(nsmc_tr_temp)
        ###### shuffled!
        nsmc_tr_shuffled = [t[1] for t in nsmc_tr_temp]
        nsmc_tr_label= [int(t[2]) for t in nsmc_tr_temp]
        onehot = featurize_onehotonly(nsmc_tr_shuffled[:120000],kor_char,140)
        onehot_y = nsmc_tr_label[:120000]
        att_source = np.zeros((120000,hidden_con))
        model.fit([onehot,att_source],onehot_y,epochs=1,batch_size=bat_size,class_weight=cw)
        onehot = featurize_onehotonly(nsmc_tr_shuffled[120000:]+nsmc_val_data,kor_char,140)
        onehot_y = nsmc_tr_label[120000:]+nsmc_val_label
        att_source = np.zeros((30000,hidden_con))
        model.fit([onehot,att_source],onehot_y,validation_split=0.5,epochs=1,batch_size=bat_size,callbacks=callbacks_list,class_weight=cw)

validate_rnn_self_drop_onehot(nsmc_label,32,64,256,class_weights_nsmc,64,'model/tutorial_nsmc/rec_onehot_self_drop')

####### Test phase

nsmc_model_shin = load_model('model/tutorial_nsmc/rec_shin-49-0.8227-0.8222.hdf5')
nsmc_model_cho  = load_model('model/tutorial_nsmc/rec_cho-49-0.8268-0.8264.hdf5')
nsmc_model_onehot = load_model('model/tutorial_nsmc/rec_onehot-01-0.8315-0.8312.hdf5')
nsmc_model      = load_model('model/tutorial_nsmc/rec-32-0.8363-0.8362.hdf5')
nsmc_model_char = load_model('model/tutorial_nsmc/rec_char-45-0.8336-0.8334.hdf5')

nsmc_self_model_shin = load_model('model/tutorial_nsmc/rec_self_shin-47-0.8335-0.8335.hdf5')
nsmc_self_model_cho  = load_model('model/tutorial_nsmc/rec_self_cho-45-0.8389-0.8388.hdf5')
nsmc_self_model_onehot = load_model('model/tutorial_nsmc/rec_onehot_self_drop-01-0.8347-0.8345.hdf5')
nsmc_self_model      = load_model('model/tutorial_nsmc/rec_self_char_dense-19-0.8445-0.8442.hdf5')
nsmc_self_model_char = load_model('model/tutorial_nsmc/rec_self_char-43-0.8374-0.8373.hdf5')

nsmc_test_shin, nsmc_test_cho, nsmc_test_onehot, nsmc_test, nsmc_test_char = featurize_rnnchar(nsmc_test_data, 100, kor_char, 140)

test_bilstm(nsmc_model_shin, nsmc_test_shin, nsmc_test_label)
test_bilstm(nsmc_model_cho, nsmc_test_cho, nsmc_test_label)
test_bilstm(nsmc_model_onehot, nsmc_test_onehot, nsmc_test_label)
test_bilstm(nsmc_model, nsmc_test, nsmc_test_label)
test_bilstm(nsmc_model_char, nsmc_test_char, nsmc_test_label)

test_bilstm_self(nsmc_self_model_shin, 64, nsmc_test_shin, nsmc_test_label)
test_bilstm_self(nsmc_self_model_cho, 64, nsmc_test_cho, nsmc_test_label)
test_bilstm_self(nsmc_self_model_onehot, 64, nsmc_test_onehot, nsmc_test_label)
test_bilstm_self(nsmc_self_model, 64, nsmc_test, nsmc_test_label)
test_bilstm_self(nsmc_self_model_char, 64, nsmc_test_char, nsmc_test_label)
