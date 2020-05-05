import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
import numpy as np
early_stopping_monitor = EarlyStopping(patience=3)

#read in training data
train_df_2 = pd.read_csv('./data/diabetes_data.csv')
test_df_2 = train_df_2.head(4)

#we drop the first line to use it for testing at the end
train_df_2 = train_df_2.iloc[4:]

#view data structure
print(train_df_2.head())

#create a dataframe with all training data except the target column
train_X_2 = train_df_2.drop(columns=['diabetes'])
test_X_2 = test_df_2.drop(columns=['diabetes'])

#check that the target variable has been removed
print(train_X_2.head())

from keras.utils import to_categorical
#one-hot encode target column
train_y_2 = to_categorical(train_df_2.diabetes)
test_y_2 = to_categorical(test_df_2.diabetes)

#vcheck that target column has been converted
print(train_y_2[0:5])

#create model
model_2 = Sequential()

#get number of columns in training data
n_cols_2 = train_X_2.shape[1]

#add layers to model
model_2.add(Dense(250, activation='relu', input_shape=(n_cols_2,)))
model_2.add(Dense(250, activation='relu'))
model_2.add(Dense(250, activation='relu'))
model_2.add(Dense(2, activation='softmax'))

#compile model using accuracy to measure model performance
model_2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#train model
#model_2.fit(train_X_2, train_y_2, epochs=300, validation_split=0.2, callbacks=[early_stopping_monitor])
#model_2.fit(train_X_2, train_y_2, epochs=300, callbacks=[early_stopping_monitor], validation_data=(test_X_2, test_y_2), batch_size=10)
model_2.fit(train_X_2, train_y_2, epochs=300, callbacks=[early_stopping_monitor], batch_size=10)

#test the model

Xnew1 = np.array([[6,148,72,35,0,33.6,0.627,50]])
ynew1 = model_2.predict_classes(Xnew1)
print(ynew1)
print("real value ","1")


Xnew2 = np.array([[1,85,66,29,0,26.6,0.35100000000000003,31]])
ynew2 = model_2.predict_classes(Xnew2)
print(ynew2)
print("real value ","0")

Xnew3 = np.array([[8,183,64,0,0,23.3,0.672,32]])
ynew3 = model_2.predict_classes(Xnew3)
print(ynew3)
print("real value ","1")

Xnew4 = np.array([[1,89,66,23,94,28.1,0.16699999999999998,21]])
ynew4 = model_2.predict_classes(Xnew4)
print(ynew4)
print("real value ","0")
