import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

import numpy as np


from sklearn.metrics import precision_recall_curve

data_path=(r'C:\Users\hlltk\PycharmProjects\churn_analysis\churn_data_proce.csv')
df = pd.read_csv(data_path)
del df['ıd']
print(df.shape)

count_no_churn = (df['Churn'] == 0).sum()
print("Churn : NO:",count_no_churn)
count_yes_churn = (df['Churn']==1).sum()
print("Churn : YES:",count_yes_churn)
pct_of_no_churn = count_no_churn/(count_no_churn+count_yes_churn)
print("Churn : NO", pct_of_no_churn*100)
pct_of_yes_churn = count_yes_churn/(count_no_churn+count_yes_churn)
print("Churn : YES:", pct_of_yes_churn*100)

print("churnnn",df['Churn'])

from sklearn.model_selection import train_test_split
X = df.loc[:, df.columns != 'Churn']
print(X.info())
y = df.loc[:, df.columns == 'Churn']



import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2, k=40)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features


import pandas as pd
import numpy as np
# import seaborn as sns
# corrmat = df.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(20,20))
# #plot heat map
# g=sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")
# plt.show()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)





sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#
# models = [('knn', KNN),
#           ('logistic', LogisticRegression),
#           ('tree', DecisionTreeClassifier),
#           ('forest', RandomForestClassifier)
#          ]
#
# param_choices = [
#     {
#         'n_neighbors': range(1, 12)
#     },
#     {
#         'C': np.logspace(-3,6, 12),
#         'penalty': ['l1', 'l2']
#     },
#     {
#         'max_depth': [1,2,3,4,5],
#         'min_samples_leaf': [3,6,10]
#     },
#     {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [1,2,3,4,5],
#         'min_samples_leaf': [3,6,10]
#     }
# ]
# #discrip
# grids = {}
# for model_info, params in zip(models, param_choices):
#     name, model = model_info
#     grid = GridSearchCV(model(), params)
#     grid.fit(X_train_res, y_train_res)
#     s = f"{name}: best score: {grid.best_score_}"
#     print(s)
#     grids[name] = grid
#
# param_grid = {
#     'n_estimators': [50, 100, 200],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [2,4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
# }
#



from sklearn.linear_model import LogisticRegression
grid={"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}
threshold = 0.4
lr=LogisticRegression(random_state=42)

lr = GridSearchCV(estimator=lr, param_grid=grid, cv= 5)
lr.fit(X_train_res, y_train_res)
print("---")
print(lr.best_params_)#en iyi parametreler

lr_best=LogisticRegression(C=1.0,penalty='l1')
print(y_test[:10])
print(lr_best.fit(X_train_res, y_train_res))
#print(lr_best.predict(X_test)[:10])

print(lr_best.predict_proba(X_test)[:10])
y_pred_lr = (lr_best.predict_proba(X_test)[:,1] >= 0).astype(int)
#print((lr_best.predict_proba(X_test)[:,1] >= 0)[:10]).astpye(int)
print(y_pred_lr[:10])
from sklearn.metrics import confusion_matrix,accuracy_score
accuracy = accuracy_score(y_test, y_pred_lr)
print(round(accuracy,4,)*100, "%")
confusion_matrix_lr = confusion_matrix(y_test, y_pred_lr)
print(confusion_matrix_lr)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_lr))



from sklearn.ensemble import RandomForestClassifier
threshold = 0.4
rfc=RandomForestClassifier(random_state=42)

rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
rfc.fit(X_train_res, y_train_res)

print(rfc.best_params_)#en iyi parametreler

rfc_best=RandomForestClassifier(random_state=42, max_features='auto', n_estimators= 50, max_depth=8, criterion='gini')
print(rfc_best.fit(X_train_res, y_train_res))
y_pred_rfc = (rfc_best.predict_proba(X_test)[:,1] >= 0.6).astype(bool)
from sklearn.metrics import confusion_matrix,accuracy_score
accuracy = accuracy_score(y_test, y_pred_rfc)
print(round(accuracy,4,)*100, "%")




confusion_matrix_forest = confusion_matrix(y_test, y_pred_rfc)
print(confusion_matrix_forest)
import seaborn as sns

#plotting a confusion matrix
labels = ['Not Churned', 'Churned']
plt.figure(figsize=(7,5))
ax= plt.subplot()
sns.heatmap(confusion_matrix_forest,cmap="Blues",annot=True,fmt='.1f', ax = ax)
plt.show()
# labels, title and ticks
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix Random Forests')

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_rfc))
