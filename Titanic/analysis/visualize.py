import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns  
from sklearn import preprocessing as pp

train_df=pd.read_csv('../data/train.csv')
test_df=pd.read_csv('../data/test.csv')

pclass=train_df[['Pclass','Survived']].groupby(['Pclass']).mean().sort_values(by='Survived',ascending=False)
sex=train_df[['Sex','Survived']].groupby(['Sex']).mean().sort_values(by='Survived',ascending=False)
sibsp=train_df[['SibSp','Survived']].groupby(['SibSp']).mean().sort_values(by='Survived',ascending=False)
parch=train_df[['Parch','Survived']].groupby(['Parch']).mean().sort_values(by='Survived',ascending=False)


print(pclass)
print('-'*30)
print(sex)
print('-'*30)
print(sibsp)
print('-'*30)
print(parch)


g=sns.FacetGrid(train_df,col='Survived',row='Pclass')
g.map(plt.hist,'Age',bins=20)

grid = sns.FacetGrid(train_df, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()


print('Shape :',train_df.shape)
train_df=train_df.drop(['Ticket','Cabin'],axis=1)
print("Shape :",train_df.shape)


train_df['Title']=train_df.Name.str.extract('([A-Za-z]+)\.',expand=False)


cross=pd.crosstab(train_df['Title'],train_df['Sex'])
print('-'*30)
print(cross)


train_df=train_df.drop(['Name','PassengerId'],axis=1)
print('-'*30)
print(train_df.columns)


def convertToNumerical(col,fitted):
	label=pp.LabelEncoder()
	label.fit(fitted)
	return label.transform(train_df[:,col])


# train_df[:,9]=convertToNumerical(9,['Capt','Col','Countess','Don','Dr','Jonkheer','Lady','Major','Master','Miss','Mlle','Mme','Mr','Mrs','Ms','Rev','Sir'])

title=train_df[['Title','Sex','Survived']].groupby(['Title','Sex']).mean().sort_values(by='Survived',ascending=False)
print('-'*30)
print(title)

print(train_df.Embarked.value_counts())

