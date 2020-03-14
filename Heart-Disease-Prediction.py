#!/usr/bin/env python
# coding: utf-8

# In[214]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df=pd.read_csv('C:/Users/sahni/Downloads/processed.cleveland.data')

df.columns=['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','num']


# In[309]:


indexNames = df[df['ca'] == '?'].index
df.drop(indexNames, inplace=True)

indexNames = df[df['thal'] == '?'].index
df.drop(indexNames, inplace=True)

df.info()
num = pd.get_dummies(df['num'])
num.head(5)
df.replace(to_replace = 1, value = 1, inplace = True)
df.replace(to_replace = 2, value = 1, inplace = True)
df.replace(to_replace = 3, value = 1, inplace = True)
df.replace(to_replace = 4, value = 1, inplace = True)


# In[216]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(df)
X_scaled=scaler.transform(df)
print(df)


# In[310]:


from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
pca.fit(X_scaled)
X_pca=pca.transform(X_scaled)

ex_variance=np.var(X_pca,axis=0)
ex_variance_ratio = ex_variance/np.sum(ex_variance)
print(ex_variance_ratio)
print(X_pca.shape)


# In[242]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
Xax=X_pca[:,0]
Yax=X_pca[:,1]
#Yaxy=X_pca[:,2]

plt.figure(figsize=(8,7))
plt.scatter(Xax, Yax, c=num[1])
plt.xlabel("First Component")
plt.ylabel("Second Component")
plt.show()


# In[243]:


plt.matshow(pca.components_,cmap='viridis')
plt.yticks([0,1,2],['1st Comp','2nd Comp','3rd Comp'],fontsize=10)
plt.colorbar()
plt.xticks(range(len(df.columns)),df.columns,ha='left')
plt.show()


# In[312]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

standardScaler = StandardScaler()
columns_to_scale = ['age','trestbps','chol','thalach','oldpeak']
df[columns_to_scale] = standardScaler.fit_transform(df[columns_to_scale])

y = df['num']
X = df.drop(['num'],axis=1)
print(df)


# In[313]:


from sklearn.linear_model import LogisticRegression
regr = LogisticRegression()

Xp = df.drop("num",axis=1)
Yt = df["num"]


# In[314]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(Xp,Yt,test_size=0.30,random_state=0)

regr.fit(X_train,Y_train)


# In[315]:


model = regr.predict(X_test)


# In[316]:


from sklearn.metrics import confusion_matrix

confusion_matrix(Y_test,model)


# In[317]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,model)


# In[296]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn_scores = []
for k in range (1,15):
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    score=cross_val_score(knn_classifier, x, y, cv=10)
    knn_scores.append(score.mean())


# In[294]:


plt.plot([k for k in range(1,15)], knn_scores, color = 'green')
for i in range(1,15):
    plt.text(i, knn_scores[i-1], (i, knn_scores[i-1]))
plt.xticks([i for i in range(1,15)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')


# In[299]:


knn_classifier = KNeighborsClassifier(n_neighbors=12)
score=cross_val_score(knn_classifier, x, y, cv=10)


# In[300]:


score.mean()


# In[239]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
model = clf.fit(X,y)


# In[284]:


randomforest_classifier = RandomForestClassifier(n_estimators=50)
score=cross_val_score(randomforest_classifier, X, y, cv=10)
score.mean()

