import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

data_train = pd.read_csv("drebin-215-dataset-5560malware-9476-benign.csv")
data_test = pd.read_csv("malgenome-215-dataset-1260malware-2539-benign.csv")
data_train = data_train.replace('?', np.nan)
data_train = data_train.dropna()
# data_test = data_test.replace('?', np.nan)
# data_test = data_test.dropna()
Y_train = data_train['class']
Y_train = np.array([str(i) for i in Y_train])
X_train = data_train.drop('class', 1)
# Y_test = data_test['class']
# Y_test = np.array([str(i) for i in Y_test])
# X_test = data_test.drop('class', 1)

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=0.33, random_state=42)

learning_rate = 0.01
gb = GradientBoostingClassifier(n_estimators=150, learning_rate=learning_rate, max_features=200, max_depth=200, random_state=0)
gb.fit(X_train, Y_train)
print("Learning rate: ", learning_rate)
print("Accuracy score (training): {0:.3f}".format(gb.score(X_train, Y_train)))
print("Accuracy score (test): {0:.3f}".format(gb.score(X_test, Y_test)))

new_label = gb.predict(X_test)
print(classification_report(new_label, Y_test, digits=4))
