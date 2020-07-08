#%%

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from pandas.plotting import register_matplotlib_converters

%matplotlib inline
%config InlineBackend.figure_format='retina'

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 30, 15

np.random.seed(101)
tf.random.set_seed(101)

path = r'C:\Users\jmruizr\Downloads\^GSPC (3).csv'
data = pd.read_csv(path, parse_dates=['Date'], index_col='Date')
df = data['Close']
df = pd.DataFrame(df) 

#%%
class AnomalyDetection():

    def __init__(self,dataset):
        self._dataset = dataset

    def fillna(self, metodo):
        self._dataset = self._dataset.fillna(method= metodo)

    def first_plot(self,etiqueta,colore,titulo):

        plt.plot(self._dataset, label=etiqueta,color=colore)
        plt.title(titulo)
        plt.legend()
    
    def split_train_test(self,partition):

        train_sz = int(len(self._dataset) * partition)
        test_sz = len(self._dataset) - train_sz
        train, test = self._dataset.iloc[0:train_sz], df.iloc[train_sz:len(self._dataset)]
        print(train.shape, test.shape)
        
        return train,test
    
    def standar_scaler(self,train,test):
                
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler = scaler.fit(train[['Close']])
        train['Close'] = scaler.transform(train[['Close']])
        test['Close'] = scaler.transform(test[['Close']])

        return train, test, scaler

    def create_dataset(self,X, y, time_steps=1):

        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = X.iloc[i:(i + time_steps)].values
            Xs.append(v)        
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    def model_arquitecture(self,unids,X_train):

        model = keras.Sequential()
        model.add(keras.layers.LSTM(
            units=64, 
            input_shape=(X_train.shape[1], X_train.shape[2])
        ))
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
        model.add(keras.layers.LSTM(units=64, return_sequences=True))
        model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
        model.compile(loss='mae', optimizer='Adam')
        return model  

    def plot_results(self,history):
                    
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()

# %%

A = AnomalyDetection(df)
A.fillna('ffill')
A.first_plot('close price','blue','S&P 500')
trai_, test_ = A.split_train_test(0.90)
train_sc, test_sc, scaler = A.standar_scaler(trai_,test_)

#%%
train_X, train_y = A.create_dataset(train_sc[['Close']], train_sc.Close,30)
test_X, test_y = A.create_dataset(test_sc[['Close']], test_sc.Close,30)
model = A.model_arquitecture(64,train_X) 
history = model.fit(train_X, train_y, epochs=10,batch_size=32,validation_split=0.1,shuffle=False)
A.plot_results(history)
train_X_prediction = model.predict(train_X)
train_mae_loss = np.mean(np.abs(train_X_prediction - train_X), axis=1)
sns.distplot(train_mae_loss, bins=50, kde=True);
test_X_prediction = model.predict(test_X)
test_mae_loss = np.mean(np.abs(test_X_prediction - test_X), axis=1)
#%%
Anomaly_score = 0.75 #Threshold

test_score = pd.DataFrame(index=test_sc[30:].index)
test_score['loss'] = test_mae_loss
test_score['threshold'] = Anomaly_score
test_score['anomaly'] = test_score.loss > test_score.threshold
test_score['Close'] = test_sc[30:].Close
# %%
plt.plot(test_score.index, test_score.loss, label='loss')
plt.plot(test_score.index, test_score.threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend();

# %%
anomalies = test_score[test_score.anomaly == True]
anomalies.head()

# %%

plt.plot(
  test_sc[30:].index, 
  scaler.inverse_transform(test_sc[30:].Close), 
  label='close price'
);

sns.scatterplot(
  anomalies.index,
  scaler.inverse_transform(anomalies.Close),
  color=sns.color_palette()[3],
  s=52,
  label='anomaly'
)
plt.xticks(rotation=25)
plt.legend();

# %%
