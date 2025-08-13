import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

train_data = pd.read_csv('train.csv')
print(train_data.isnull().sum())

train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
train_data['Embarked'].fillna(train_data['Embarked'].mode()[0], inplace=True)

features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']
X=train_data[features]
y=train_data['Survived']

X=pd.get_dummies(X, columns=['Sex','Embarked'], drop_first=True)

scaler = StandardScaler()
X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
train_accuracy = model.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Train_accuracy:{train_accuracy:.2f}')
print(f'Test_accuracy:{test_accuracy:.2f}')

print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('titan_confusion_matrix.png')
plt.show()

features_importance = pd.Series(model.feature_importances_, index=X.columns)
plt.figure(figsize=(8, 6))
features_importance.sort_values().plot(kind='barh')
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
plt.show()