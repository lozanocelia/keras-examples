import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

#read in training data
train_df = pd.read_csv('./data/hourly_wages_data.csv')

#create a dataframe with all training data except the target column
train_X = train_df.drop(columns=['wage_per_hour'])


#view data structure
print(train_df.head())

#create a dataframe with only the target column
train_y = train_df[['wage_per_hour']]

#view dataframe
print(train_y.head())

#create model
model = Sequential()

#get number of columns in training data
n_cols = train_X.shape[1]

#add model layers
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(10, activation='relu'))
model.add(Dense(1))

#compile model using mse as a measure of model performance
model.compile(optimizer='adam', loss='mean_squared_error')

#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)

#train model
model.fit(train_X, train_y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])