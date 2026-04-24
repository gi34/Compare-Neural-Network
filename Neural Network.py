import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('diabetes.csv')

#standardization
X = df.iloc[:, :-1]
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


#splitting data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_scaled_df = pd.DataFrame(X_train, columns=X.columns)

#modelling
perceptron = Perceptron(random_state=42,
                        penalty='l1',
                        alpha=0.001,
                        max_iter=1000,
                        eta0=1.0)

perceptron.fit(X_train,y_train)
y_pred = perceptron.predict(X_test)
y_pred_perceptron = perceptron.decision_function(X_test)

#logistic regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred2 = log_reg.predict(X_test)

print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred2))

#ROC
y_pred_proba = log_reg.predict_proba(X_test)[::,-1]
fpr, tpr, _ = metrics.roc_curve(y_test,y_pred_perceptron)
auc = metrics.roc_auc_score(y_test, y_pred_perceptron)

plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

#evaluating
accuracy = accuracy_score(y_test,y_pred)
print(f'Accuracy: {accuracy:.2f}\n')
print(classification_report(y_test,y_pred))
print('Confusion Matrix:\n')
print(confusion_matrix(y_test,y_pred))


'''
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
#X_scaled_df = pd.DataFrame(X_scaled,columns=X.columns)
#print(X_scaled_df)

#decision boundary
plt.subplot(1, 2, 1)
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdBu)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.RdBu)
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.title('Perceptron Decision Boundary')
plt.colorbar()

selected_columns=['Outcome','Pregnancies','BloodPressure','SkinThickness','Insulin','DiabetesPedigreeFunction','BMI','Age','Glucose']
sns.pairplot(df[selected_columns],hue='Outcome')
plt.show()

selected_columns=['Pregnancies','BloodPressure','SkinThickness','Insulin','DiabetesPedigreeFunction','BMI','Age','Glucose']
corr=df[selected_columns].corr()
print(corr['Outcome'].sort_values(ascending=False))

sns.heatmap(corr,annot=True, cmap='coolwarm',fmt='.2f')
plt.title("Correlation Heatmap of Attributes")
plt.show()

#histogram
pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
sns.histplot(df,x='Glucose',hue='Outcome')
plt.show()
sns.histplot(df,x='Pregnancies',hue='Outcome')
plt.show()
sns.histplot(df,x='BloodPressure',hue='Outcome')
plt.show()
sns.histplot(df,x='SkinThickness',hue='Outcome')
plt.show()
sns.histplot(df,x='Insulin',hue='Outcome')
plt.show()
sns.histplot(df,x='DiabetesPedigreeFunction',hue='Outcome')
plt.show()
sns.histplot(df,x='BMI',hue='Outcome')
plt.show()
sns.histplot(df,x='Age',hue='Outcome')
plt.show()
sns.histplot(df,x='Outcome',hue='Outcome')
plt.show()



selected_columns=['Pregnancies','BMI','Age','Glucose']
second_columns=['SkinThickness','BloodPressure','Insulin']
third_columns=['DiabetesPedigreeFunction']
sns.boxplot(data=df[selected_columns],orient='h')
plt.title('Boxplot of attributes')
plt.show()

#standardization

X = df.iloc[:, :-1]
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled,columns=X.columns)
print(X_scaled_df)

#splitting data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_scaled_df = pd.DataFrame(X_train, columns=X.columns)
print(X_train_scaled_df)
'''


#selected_columns=['Pregnancies','Glucose','SkinThickness','Age','Outcome','BloodPressure','DiabetesPedigreeFunction','BMI','Insulin']

'''
#outlier
Q1=df.quantile(0.25)
Q3 = df.quantile(0.75)

IQR=Q3-Q1
Upper = Q3+(1.5*IQR)
Lower = Q1-(1.5*IQR)

cleaned_df = df.copy()
for column in df.columns:
    cleaned_df = cleaned_df[(cleaned_df[column] >= Lower[column]) & (cleaned_df[column] <= Upper[column])]
print(cleaned_df)
'''
'''
glucose = df['Glucose']
age = df['Age']
outcome = df['Outcome']

plt.xlabel('BMI')
plt.ylabel('Age')

plt.scatter(glucose[outcome == 0], age[outcome == 0], label='Outcome 0', color='blue')
# Plotting the scatter plot for Outcome = 1
plt.scatter(glucose[outcome == 1], age[outcome == 1], label='Outcome 1', color='red')
plt.title('Scatter Plot of Glucose vs Age')
plt.legend()
plt.show()

sns.histplot(df,x=df['Outcome'])
plt.show()
'''









'''
sns.scatterplot(df, x=df["age"],y=df["bmi"],hue=df["diabetes"])
plt.show()

sns.histplot(df,x = df['Glucose'],hue = 'Outcome')
plt.show()

selected_columns=['bmi','hypertension','heart_disease','HbA1c_level','blood_glucose_level','diabetes','age']
sns.pairplot(df[selected_columns])
plt.show()


selected_columns=['bmi','hypertension','heart_disease','HbA1c_level','blood_glucose_level','diabetes','age']
corr = df[selected_columns].corr()
print(corr["diabetes"].sort_values(ascending=False))

sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Selected Columns')
plt.show()
'''

'''
selected_columns=['smoking_history','hypertension','blood_glucose_level']
plt.xlabel('blood_glucose_level')
plt.ylabel('smoking_history')
plt.scatter('blood_glucose_level','hypertension', label='blood', color='red')
#plt.scatter('smoking_history','diabetes', label='smoking', color='green')
plt.show()
'''



'''
Q1=boost.quantile(0.25)
Q3 = boost.quantile(0.75)

IQR=Q3-Q1
Upper = Q3+(1.5*IQR)
Lower = Q1-(1.5*IQR)
print('Upper fence: '+ str(Upper))
print('Lower fence: '+ str(Lower))
'''



