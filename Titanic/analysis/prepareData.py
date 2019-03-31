import pandas as pd
import numpy as np

train_df=pd.read_csv('../data/train.csv')
test_df=pd.read_csv('../data/test.csv')


# # Catagorical 
# train_df['Title']=train_df.Name.str.extract('([A-Za-z]+)\.',expand=False)

# # Numerical
# train_df['Family']=train_df['SibSp']+train_df['Parch'];


# #Drop
# train_df=train_df.drop(['PassengerId','Cabin','Ticket'],axis=1)
# train_df=train_df.drop(['SibSp','Parch'],axis=1) 
# train_df=train_df.drop(['Name'],axis=1)


# #title vs age
# tvsa=train_df[['Title','Age']].groupby(['Title']).mean().sort_values(by='Age')
# # Embarked vs survived
# evss=train_df[['Embarked','Survived']].groupby(['Embarked']).mean().sort_values(by='Survived')


# title=train_df.Title.unique()
# guese_age={}

# for t in title:
# 	mean_age=train_df[train_df['Title']==t]['Age'].dropna()
# 	guese=mean_age.mean()
# 	guese_age[t]=guese;

# for t in title:
# 	train_df.loc[ (train_df['Title']==t) & (train_df['Age'].isnull()),'Age']=guese_age[t]


# train_df.to_csv('../data/processedTrain.csv',encoding='utf-8')

combine=[train_df,test_df]
fileName=['process_train.csv','process_test.csv']
i=0

for dataset in combine:
    #categorical
    dataset['Title']=dataset.Name.str.extract('([A-Za-z]+)\.',expand=False)
   
    # Numerical
    dataset['Family']=dataset['SibSp']+dataset['Parch'];

    
    title=dataset.Title.unique()
    guese_age=[]
    
    dataset['Fare'].fillna(dataset['Fare'].dropna().median(), inplace=True)

    for t in title:
    	mean_age=dataset[dataset['Title']==t]['Age'].dropna()
        guese_age.append(mean_age.mean())
    
    j=0
    for t in title:
    	dataset.loc[ (dataset.Title==t) & (dataset.Age.isnull()),'Age']=guese_age[j]
    	j+=1
    
    dataset['Age'].fillna(dataset['Age'].dropna().mean(),inplace=True)

    # [u'Pclass', u'Sex', u'Age', u'Fare', u'Embarked', u'Title', u'Family']
    dataset.loc[(dataset['Embarked'].isnull()),'Embarked']='S'
    
    #Drop
    dataset=dataset.drop(['PassengerId','Cabin','Ticket'],axis=1)
    dataset=dataset.drop(['SibSp','Parch'],axis=1) 
    dataset=dataset.drop(['Name'],axis=1)


    # print(dataset['Title'].dropna())
    dataset.to_csv('../data/'+fileName[i],encoding='utf-8')
    i+=1


