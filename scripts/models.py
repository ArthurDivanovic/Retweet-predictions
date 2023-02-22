import csv
import numpy as np
import pandas as pd
import tensorflow as tf
# Deep learning: 
from tensorflow.keras.layers import Input
from keras.models import Model
from keras.layers import LSTM, Dense, Embedding, concatenate, Dropout, Concatenate, concatenate
from keras.layers import Bidirectional
import numpy as np
from sklearn.metrics import mean_absolute_error
# Loading the word tokenizer
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier

"""

RNN model with concatenation of pre-trained layers for embedding and layers with the numerical features
No need to tokenize text before 
"""


class TextToTensor():

    def __init__(self, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len

    def string_to_tensor(self, string_list: list) -> list:
        """
        A method to convert a string list to a tensor for a deep learning model
        """    
        string_list = self.tokenizer.texts_to_sequences(string_list)
        string_list = pad_sequences(string_list, maxlen=self.max_len)
        
        return string_list


class Embeddings():
    """
    A class to read the word embedding file and to create the word embedding matrix
    """

    def __init__(self, path, vector_dimension):
        self.path = path 
        self.vector_dimension = vector_dimension
    
    @staticmethod
    def get_coefs(word, *arr): 
        return word, np.asarray(arr, dtype='float32')

    def get_embedding_index(self):
        embeddings_index = dict(self.get_coefs(*o.split(" ")) for o in open(self.path, errors='ignore'))
        return embeddings_index

    def create_embedding_matrix(self, tokenizer, max_features):
        """
        A method to create the embedding matrix
        """
        model_embed = self.get_embedding_index()

        embedding_matrix = np.zeros((max_features + 1, self.vector_dimension))
        for word, index in tokenizer.word_index.items():
            if index > max_features:
                break
            else:
                try:
                    embedding_matrix[index] = model_embed[word]
                except:
                    continue
        return embedding_matrix


    # Creating the embedding matrix
# Tokenizing the text  
def create_embed(texts,embed_path,embed_dim):

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(texts)
        embedding = Embeddings(embed_path, embed_dim)
        embedding_matrix = embedding.create_embedding_matrix(tokenizer, len(tokenizer.word_counts))
        return embedding_matrix



class RnnModel():
    """
    A recurrent neural network for semantic analysis
    """

    def __init__(self, embedding_matrix, embedding_dim, max_len, nbre_feats, X_additional=None):
        


  
        inp1 = Input(shape=(max_len,))
        inp2 = Input(shape=(nbre_feats,))
        feats = Dense(128, activation = "relu")(inp2)
        
        x = Embedding(embedding_matrix.shape[0], embedding_dim, weights=[embedding_matrix])(inp1)
        x = Bidirectional(LSTM(256, return_sequences=True))(x)
        x = Bidirectional(LSTM(150))(x)
        x = Dense(128, activation="relu")(x)
        x = Dropout(0.1)(x)
        
        concat = concatenate([x, feats], name='Concatenate', axis=1)

        print(type(concat))
        x = Dense(64, activation="relu")(concat)
        final_output = Dense(1, activation="relu")(x)    
        model = Model(inputs=[inp1,inp2], outputs=final_output, name='Final_output')

        model.compile(loss = 'mae', optimizer = 'rmsprop')
        self.model = model



class Pipeline:
    """
    A class for the machine learning pipeline
    """
    def __init__(
        self, 
        X_train_duo: list, 
        Y_train: list, 
        embedding_matrix: np.array,
        X_test=[], 
        Y_test=[],
        epochs=3,
        batch_size=256
        ):
        embed_dim = 200

        X_train = X_train_duo['text']
        feats = feat_creation(X_train_duo).drop(columns=['retweets_count'])
        nbre_feats = len(feats.axes[1])
        
        # Preprocecing the text : ajouter clean data ici
        Y_train = np.asarray(Y_train)
        
        # Tokenizing the text
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(X_train)

        
        # Creating the padded input for the deep learning model
        max_len = np.max([len(text.split()) for text in X_train])
        TextToTensor_instance = TextToTensor(
            tokenizer=tokenizer, 
            max_len=max_len
            )
        X_train = TextToTensor_instance.string_to_tensor(X_train)
        
        X_feat = tf.constant(feats.to_numpy(),dtype='float64')
        

        # Creating the model
        rnn = RnnModel(
            embedding_matrix=embedding_matrix, 
            embedding_dim=embed_dim, 
            max_len=max_len,
            nbre_feats = nbre_feats,
        )
        rnn.model.fit(
            [X_train,X_feat],
            Y_train, 
            batch_size=batch_size, 
            epochs=epochs
        )

        self.model = rnn.model

        # If X_test is provided we make predictions with the created model
        if len(X_test)>0:
            
            X_test_text = X_test['text']
            X_test_feats = feat_creation(X_test)
            X_test = TextToTensor_instance.string_to_tensor(X_test_text)
            pred = rnn.model.predict([X_test_text,X_test_feats]).tolist()
            self.pred = pred


def embed_rnn_fit(X_train_duo, Y_train, epochs, batch_size, X_test) :
    embedding_matrix = create_embed(X_train_duo['text'],embed_path= 'embeddings/glove.twitter.27B.200d.txt', embed_dim=200)
    return Pipeline(
X_train_duo= X_train_duo,
Y_train=Y_train,
embedding_matrix = embedding_matrix,
X_test = X_test,
epochs= epochs,
batch_size=batch_size
)


"""

Classifier model then regression model on each classes

"""


def model_class_to_reg(X_train1,X_test1, y_train1, y_test1, X_eval1, eval_data):
    
    #creating the labels
    X_train,X_test, y_train, y_test, X_eval = X_train1.copy(),X_test1.copy(), y_train1.copy(), y_test1.copy(), X_eval1.copy()
    y_train = pd.DataFrame(y_train, columns=['retweets_count'])
    y_test = pd.DataFrame(y_test, columns=['retweets_count'])
    y_train['label'] = np.zeros(len(y_train))
    y_train[y_train['retweets_count']>0]['label'] = pd.qcut( y_train[y_train['retweets_count']>0]['retweets_count'], 3, labels = False)+1
    
    
    
    
    #Classify in 4 categories
    clf=RandomForestClassifier()
    clf.fit(X_train,y_train['label'])

    #Predict the categories
    train_label_pred = clf.predict(X_train)
    X_test['label'] = clf.predict(X_test)
    X_eval['label'] = clf.predict(X_eval)
    X_train['label'] = y_train['label']
    X_eval =  pd.concat([X_eval,eval_data['TweetID']],axis=1)

    # Sort to get the same order between X and Y_pred
    y_test['label'] = X_test['label']
    y_test = y_test.sort_values(by=[ 'label'])
    X_eval = X_eval.sort_values(by=['label'])
    X_train = X_train.sort_values(by=['label'])
    X_test = X_test.sort_values(by=['label'])
    
    pred_test = []
    pred_train = []
    pred_eval = []
    
    # regression on each label
    for i in range(4):
        model=xgb.XGBRegressor(eval_metric = 'mae')
        model.fit(X_train[X_train['label']==i],y_train['retweets_count'][y_train['label']==i])
        pred_train.extend(model.predict(X_train[X_train['label']==i]))
        pred_test.extend(model.predict(X_test[X_test['label']==i]))
        pred_eval.extend(model.predict(X_eval[X_eval['label']==i].drop(columns=['TweetID'])))
    
    X_eval['results'] = pred_eval
    
    print('MAE sur test:',mean_absolute_error(y_test['retweets_count'],pred_test))
    print('MAE sur train:',mean_absolute_error(y_train['retweets_count'],pred_train),'\n')
    
    return X_eval, y_test, pred_test, y_train, pred_train 





"""

To export results

"""

def export_function(y_eval, eval_data,titre_model):
    result = pd.DataFrame(columns=['TweetID','retweets_count'])
    result['TweetID'] = eval_data['TweetID']
    result['retweets_count'] = y_eval.astype(int)
    result.to_csv("eval_files/Prediction_"+titre_model+".csv", index=False)
