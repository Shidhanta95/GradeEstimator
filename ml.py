# Load libraries
from sklearn.ensemble import AdaBoostClassifier
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pandas as pd

df = pd.read_csv('data_mod.csv')



print(df.head())


df = df.dropna()

y = df.loc[:,'Class']
y = pd.DataFrame(y)
X = df.drop(['Class'],axis=1)


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) 

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=150,
                          learning_rate=0.5)
# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))