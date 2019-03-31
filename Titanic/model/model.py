import pandas as pd
from sklearn import preprocessing as pp 
import numpy as np
#  machine learning
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import Perceptron
# from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

train=pd.read_csv('../data/process_train.csv')
test=pd.read_csv('../data/process_test.csv')
py=pd.read_csv('../data/gender_submission.csv')

titleList=['Mlle','Mme','Sir','Jonkheer','Lady','Major','Capt','Col','Countess','Don','Mr', 'Mrs' ,'Miss', 'Master', 'Ms' ,'Col' ,'Rev' ,'Dr', 'Dona']
titleMap=dict()

i=1
for t in titleList:
	titleMap[t]=i
	i+=1 

train['Title']=train['Title'].map(titleMap)
test['Title']=test['Title'].map(titleMap)

train_data_x=np.asanyarray(train[['Pclass','Sex','Age','Fare','Embarked','Title','Family']])
train_data_y=train['Survived'].values

test_data_x=np.asanyarray(test[['Pclass','Sex','Age','Fare','Embarked','Title','Family']])
test_data_y=np.asanyarray(py['Survived'])


def convertToNumerical(col,dataset,fitted):
	label=pp.LabelEncoder()
	label.fit(fitted)
	return label.transform(dataset[:,col])


train_data_x[:,1]=convertToNumerical(1,train_data_x,['male','female'])
train_data_x[:,4]=convertToNumerical(4,train_data_x,['S','C','Q'])

test_data_x[:,1]=convertToNumerical(1,test_data_x,['male','female'])
test_data_x[:,4]=convertToNumerical(4,test_data_x,['S','C','Q'])


# decissionTree=DecisionTreeClassifier()
# decissionTree.fit(train_data_x,train_data_y)

# predict_y=decissionTree.predict(test_data_x)

# print('score :',round(decissionTree.score(train_data_x,train_data_y)*100,2))
# print("score :",accuracy_score(test_data_y,predict_y)*100)

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_data_x,train_data_y)
predict_y=random_forest.predict(test_data_x)

# print('score :',round(decissionTree.score(train_data_x,train_data_y)*100,2))
print("score :",accuracy_score(test_data_y,predict_y)*100)


submission=pd.DataFrame({
       'PassengerId':py['PassengerId'],
       "Survived":predict_y	
	})

pd.DataFrame(submission).to_csv('../data/submission.csv',index=False,encoding='utf-8')
    




