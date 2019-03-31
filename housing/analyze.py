import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df_train=pd.read_csv('./train.csv')


sns.distplot(df_train['SalePrice'])
plt.show()


print(df_train['SalePrice'].describe())