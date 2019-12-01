import pandas as pd
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

df = pd.read_csv('drebin215dataset5560malware9476benign.csv', low_memory=False)

df.loc[df['class'] == 'S', 'class'] = 1
df.loc[df['class'] == 'B', 'class'] = 0
df = df.replace('?', np.nan)
df = df.dropna()

X = df.drop('class', axis=1)
y = df.iloc[:,-1]
df.loc[df['class'] == 'S', 'class'] = 1
df.loc[df['class'] == 'B', 'class'] = 0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

svm = LinearSVC()
svm.fit(X_train, y_train)
new_label = svm.predict(X_test)
print(classification_report(new_label, y_test, digits=4))
