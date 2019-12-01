import numpy as np
import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('drebin-215-dataset-5560malware-9476-benign.csv', low_memory=False)

df.loc[df['class'] == 'S', 'class'] = 1
df.loc[df['class'] == 'B', 'class'] = 0
df = df.replace('?', np.nan)
df = df.dropna()

X = df.drop('class', axis=1)
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = Sequential()
model.add(Dense(32, input_dim=215, activation='sigmoid'))
model.add(Dense(64, activation='sigmoid'))
model.add(Dense(32, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=20, batch_size=32)
predictions = model.predict_classes(X_test)
print(classification_report(predictions, y_test, digits=4))



