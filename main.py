import pandas as pd 
import statistics as s
from sklearn.neighbors import KNeighborsClassifier
import sklearn 


train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
train_df['Fare'].fillna(train_df['Fare'].mean(), inplace=True)
test_df['Fare'].fillna(test_df['Fare'].mean(), inplace=True)
train_df['Age'].fillna(train_df['Fare'].mean(), inplace=True)
test_df['Age'].fillna(test_df['Fare'].mean(), inplace=True)
cleanup_sex = {"Sex": {"male": 1, "female": 0}}
train_df.replace(cleanup_sex, inplace=True)
test_df.replace(cleanup_sex, inplace=True)


X_train = train_df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age']]


X_train = X_train.as_matrix(columns = X_train.columns[0:])
y_train = train_df[['Survived']]
y_train = y_train.Survived.tolist()
X_test = test_df[['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Age']]
X_test = X_test.as_matrix(columns = X_test.columns[0:])

knn = KNeighborsClassifier()
knn.fit(X = X_train, y = y_train)
predicted = knn.predict(X=X_test)

result = pd.DataFrame(list(zip(test_df['PassengerId'], predicted)), columns =['PassengerId', 'Survived'])
export_csv = result.to_csv(r'~/Yandex.Disk.localized/M/Kaggle/Titanic/submission.csv', index = None)
