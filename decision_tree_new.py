#Importing libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Input data files are available in the "../input/" directory.
import os
import matplotlib.pyplot as plt#visualization
from PIL import  Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
import seaborn as sns#visualization
import itertools
import warnings
warnings.filterwarnings("ignore")
import io
import plotly.offline as py#visualization
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
py.init_notebook_mode(connected=True)#visualization
import plotly.graph_objs as go#visualization
import plotly.tools as tls#visualization
import plotly.figure_factory as ff#visualization
from IPython import get_ipython
#from mlxtend.feature_selection import SequentialFeatureSelector as sfs

input_path = r"C:\Users\hlltk\PycharmProjects\churn_analysis"
input_filename = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


telcom = pd.read_csv(input_path+"/"+input_filename,sep=',')
telcom.head()

print ("Rows     : " ,telcom.shape[0])
print ("Columns  : " ,telcom.shape[1])
print ("\nFeatures : \n" ,telcom.columns.tolist())
print ("\nMissing values :  ", telcom.isnull().sum().values.sum())
print ("\nUnique values :  \n",telcom.nunique())


a = telcom['gender'].values
print("type",type(a))

# Toplam ücret sütununda boşlukları boş değerlerle değiştirme
telcom['TotalCharges'] = telcom["TotalCharges"].replace(" ", np.nan)


telcom = telcom[telcom["TotalCharges"].notnull()]
telcom = telcom.reset_index()[telcom.columns]

#Separating churn and non churn customers
churn     = telcom[telcom["Churn"] == "Yes"]
not_churn = telcom[telcom["Churn"] == "No"]


telcom["TotalCharges"] = telcom["TotalCharges"].astype(float)

#yes no değerlerinin sayılarını veriyor.
lab = telcom["Churn"].value_counts().keys().tolist()

val = telcom["Churn"].value_counts().values.tolist()
print('keys()',lab)
print('values',val)

# Aşağıdaki sütunlar için internet İnternet servisi yok’u Hayır olarak değiştirin.
replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies','MultipleLines']
for i in replace_cols:
    telcom[i] = telcom[i].replace({'No internet service': 'Yes'})


telcom["SeniorCitizen"] = telcom["SeniorCitizen"].replace({1: "Yes", 0: "No"})


def tenure_lab(telcom):
    if telcom["tenure"] <= 24:
        return "Tenure_0-24"
    elif (telcom["tenure"] > 24) & (telcom["tenure"] <= 48):
        return "Tenure_24-48"
    elif (telcom["tenure"] > 48) & (telcom["tenure"] <= 72):
        return "Tenure_48-72"


telcom["tenure_group"] = telcom.apply(lambda telcom: tenure_lab(telcom),axis=1)


churn = telcom[telcom["Churn"] == "Yes"]
not_churn = telcom[telcom["Churn"] == "No"]

Id_col = ['customerID']
target_col = ["Churn"]
cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
cat_cols = [x for x in cat_cols if x not in target_col]
num_cols = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


Id_col = ['customerID']

target_col = ["Churn"]

cat_cols = telcom.nunique()[telcom.nunique() < 6].keys().tolist()
cat_cols = [x for x in cat_cols if x not in target_col]

num_cols = [x for x in telcom.columns if x not in cat_cols + target_col + Id_col]

bin_cols = telcom.nunique()[telcom.nunique() == 2].keys().tolist()

multi_cols = [i for i in cat_cols if i not in bin_cols]


print('telcom',telcom['MonthlyCharges'])
le = LabelEncoder()
for i in bin_cols:
    telcom[i] = le.fit_transform(telcom[i])


telcom = pd.get_dummies(data=telcom, columns=multi_cols)
print('telcom',telcom)
import scipy.stats as ss
print("numcols",num_cols)
from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(telcom[num_cols])
scaled = pd.DataFrame(x_scaled,columns=['tenure','MonthlyCharges','TotalCharges'])
df_telcom_og = telcom.copy() #telcom copy
telcom = telcom.drop(columns=num_cols, axis=1)
print("telcominfooo",telcom.info())
telcom = telcom.merge(scaled, left_index=True, right_index=True, how="left")


# num_colss = len(num_cols)
# for i in range(num_colss):
#
#     print(i)
#     col = scaled[:i]
#     print(col)
#     col_stats = ss.describe(col)
#     print('col_stats',col_stats)




df_telcom_og = telcom.copy() #telcom copy
print(telcom.info())

X= telcom.drop(columns='Churn', axis=1)
del X['customerID']
del X['tenure']
# del X['Partner']
# del X['DeviceProtection']
# del X['OnlineBackup']
print("X parameters :" ,X.describe().T)
print('XXX',X['TotalCharges'])
y = telcom['Churn']
#X_train, X_test, y_train, y_test = train_test_split(X.astype (float), y, test_size=0.2, random_state=0)
X_copy = X.copy()
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# import statsmodels.api as stf
# lm = stf.OLS(y_train,X_train)
# model = lm.fit()
# print('satisfs')
# print(model.summary())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
sm = SMOTE(random_state=0)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)



thres = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression



for i in thres:
    grid = {'criterion':['gini','entropy'],"splitter":["best","random"],'max_depth': [1,2,3,4,5,6,7,8],'min_samples_leaf': [3,6,10,12,15],
        "max_features":["auto","sqrt","log2"]}
    dsc = DecisionTreeClassifier(random_state=42)

    dsc = GridSearchCV(estimator=dsc, param_grid=grid, cv=5)
    dsc.fit(X_train_res, y_train_res)
    print("---")
    print("best Params")
    print(dsc.best_params_)
    dsc_best = DecisionTreeClassifier(random_state=42, max_features='auto', max_depth=8,
                                      criterion='gini',splitter='best')
    dsc_best.fit(X_train_res, y_train_res)
    # y_pred_rfc=rfc_best.predict(X_test)
    a = dsc_best.predict_proba(X_test)[:, 1]
    for i in a:
        print("X_text",i)

    y_pred_dsc = (dsc_best.predict_proba(X_test)[:, 1] >= i).astype(bool)  # set threshold as 0.3
    from sklearn.metrics import confusion_matrix, accuracy_score

    accuracy = accuracy_score(y_test, y_pred_dsc)
    y_pred_dsc = pd.DataFrame(y_pred_dsc)
    y_test = pd.DataFrame(y_test)
    y_pred_dsc.to_csv('preddsc.csv')
    y_test.to_csv('y_test.csv')
    confusion_matrix_forest = confusion_matrix(y_test, y_pred_dsc)
    print(confusion_matrix_forest)
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.metrics import precision_recall_fscore_support
    accuracy = accuracy_score(y_test, y_pred_dsc)
    print('threshold:',i,round(accuracy, 4, ) * 100, "%")
    confusion_matrix_lr = confusion_matrix(y_test, y_pred_dsc)
    print(confusion_matrix_lr)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred_dsc))
    deger = precision_recall_fscore_support(y_test, y_pred_dsc, average='weighted')
    print("precisin - recall - fscore",deger)
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics

for i in thres:
    parameters = {'n_estimators':[100,200,300],'learning_rate':[1.0,2.0,4.0]}
    abc = AdaBoostClassifier()
    abc_grid = GridSearchCV(abc, parameters)
    # Train Adaboost Classifer
    model = abc_grid.fit(X_train_res, y_train_res)

    #Predict the response for test dataset
    #y_pred = model.predict(X_test)
    y_pred = (dsc_best.predict_proba(X_test)[:, 1] >= i).astype(bool)
    confusion_matrix_forest = confusion_matrix(y_test, y_pred)
    print(confusion_matrix_forest)
    print("Adaboost Accuracy:",metrics.accuracy_score(y_test, y_pred))
    report=classification_report(y_test,abc_grid.predict(X_test))

    print("Report",report)
    deger = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    print("precisin - recall - fscore",deger)

from sklearn.svm import SVC

svclassifier = SVC(kernel='rbf', degree=8)
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)
print('------------------SVM------------------------------------------')
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


#
#
for i in thres:
    rfc_best = RandomForestClassifier(random_state=42, max_features='auto', n_estimators=50, max_depth=8,
                                      criterion='gini')
    rfc_best.fit(X_train_res, y_train_res)
    # y_pred_rfc=rfc_best.predict(X_test)
    y_pred_rfc = (rfc_best.predict_proba(X_test)[:, 1] >= i).astype(bool)  # set threshold as 0.3
    from sklearn.metrics import confusion_matrix, accuracy_score

    accuracy = accuracy_score(y_test, y_pred_rfc)


    confusion_matrix_forest = confusion_matrix(y_test, y_pred_rfc)
    print(confusion_matrix_forest)
    from sklearn.metrics import confusion_matrix, accuracy_score

    accuracy = accuracy_score(y_test, y_pred_rfc)
    print(round(accuracy, 4, ) * 100, "%")
    confusion_matrix_lr = confusion_matrix(y_test, y_pred_rfc)
    print(confusion_matrix_lr)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred_rfc))
    deger_1 = precision_recall_fscore_support(y_test, y_pred_rfc, average='weighted')
    print("precisin - recall - fscore",deger_1)

from sklearn.neighbors import KNeighborsClassifier as KNN
for i in thres:
    grid = {'criterion': ['gini', 'entropy'], "splitter": ["best", "random"],
            'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],
            'min_samples_leaf': [3, 6, 10, 12, 15],
            "max_features":["auto","sqrt","log2"]}
    knn = KNN()
    knnGrid = {
        'n_neighbors': range(1, 12),
    }
    knn = GridSearchCV(estimator=knn, param_grid=knnGrid, cv=5)

    knn.fit(X_train_res, y_train_res)

    print(knn.best_params_)

    knn_best = KNN(algorithm='auto', leaf_size=30, metric='minkowski',
                     metric_params=None, n_jobs=None, n_neighbors=5, p=2,
                     weights='uniform')
    print(knn_best.fit(X_train_res, y_train_res))

    y_pred_knn = (knn_best.predict_proba(X_test)[:, 1] >= i).astype(bool)
    print(knn_best.predict_proba(X_test)[:, 1] == i)
    from sklearn.metrics import confusion_matrix, accuracy_score

    accuracy = accuracy_score(y_test, y_pred_knn)
    print(round(accuracy, 4, ) * 100, "%")

    confusion_matrix_forest = confusion_matrix(y_test, y_pred_knn)
    print(confusion_matrix_forest)
    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred_knn))
