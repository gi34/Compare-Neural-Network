import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('diabetes.csv')

#standardization
X = df.iloc[:, :-1]
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#splitting data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_scaled_df = pd.DataFrame(X_train, columns=X.columns)

#logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred2 = log_reg.predict(X_test)

print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred2))
print('Confusion matrix')
print(confusion_matrix(y_test,y_pred2))
