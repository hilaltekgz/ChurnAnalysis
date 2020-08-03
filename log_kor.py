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

# Dropping null values from total charges column which contain .15% missing data
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


print('teloom',telcom['MonthlyCharges'])
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
del X['Partner']
del X['DeviceProtection']
del X['OnlineBackup']
del X['StreamingTV']
del X['MultipleLines_Yes']
del X['MultipleLines_No phone service']
del X['InternetService_DSL']
del X['InternetService_No']
print('XXX',X['TotalCharges'])
y = telcom['Churn']
X_train, X_test, y_train, y_test = train_test_split(X.astype (float), y, test_size=0.4, random_state=0)
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
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression

thres = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
for i in thres:
    grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
    lr = LogisticRegression(random_state=42)

    lr = GridSearchCV(estimator=lr, param_grid=grid, cv=5)
    lr.fit(X_train_res, y_train_res)
    lr_best = LogisticRegression(C=1.0, penalty='l1')
    lr_best.fit(X_train_res, y_train_res)
    y_pred_lr = (lr_best.predict_proba(X_test)[:, 1] >= i).astype(bool)
    from sklearn.metrics import confusion_matrix, accuracy_score

    accuracy = accuracy_score(y_test, y_pred_lr)
    print("LogisticReg : acc",round(accuracy, 4, ) * 100, "%")
    confusion_matrix_lr = confusion_matrix(y_test, y_pred_lr)
    print("LogisticReg : ",confusion_matrix_lr)
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred_lr))
    deger = precision_recall_fscore_support(y_test, y_pred_lr, average='weighted')
    print("precisin - recall - fscore", deger)
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.cross_decomposition import PLSRegression, PLSSVD
from sklearn import model_selection
#CV
cv_10 = model_selection.KFold(n_splits=10, shuffle=True, random_state=1)


#Hata hesaplamak için döngü
RMSE = []

for i in np.arange(1, X_train.shape[1] + 1):
    print("i",i)
    pls = PLSRegression(n_components=i)
    print("pls",pls)
    score = np.sqrt(-1*cross_val_score(pls, X_train, y_train, cv=cv_10, scoring='neg_mean_squared_error').mean())
    RMSE.append(score)

#Sonuçların Görselleştirilmesi
plt.plot(np.arange(1, X_train.shape[1] + 1), np.array(RMSE), '-v', c = "r")
plt.xlabel('Bileşen Sayısı')
plt.ylabel('RMSE')
plt.title('Salary')
plt.show()


def calc_lift(x, y, clf, bins=10):
    """
    Takes input arrays and trained SkLearn Classifier and returns a Pandas
    DataFrame with the average lift generated by the model in each bin

    Parameters
    -------------------
    x:    Numpy array or Pandas Dataframe with shape = [n_samples, n_features]

    y:    A 1-d Numpy array or Pandas Series with shape = [n_samples]
          IMPORTANT: Code is only configured for binary target variable
          of 1 for success and 0 for failure

    clf:  A trained SkLearn classifier object
    bins: Number of equal sized buckets to divide observations across
          Default value is 10
    """

    # Actual Value of y


    y_actual = y
    # Predicted Probability that y = 1
    y_prob = lr.predict_proba(x)
    # Predicted Value of Y
    y_pred = lr.predict(x)
    cols = ['ACTUAL', 'PROB_POSITIVE', 'PREDICTED']
    data = [y_actual, y_prob[:, 1], y_pred]
    df = pd.DataFrame(dict(zip(cols, data)))

    # Observations where y=1
    total_positive_n = df['ACTUAL'].sum()
    # Total Observations
    total_n = df.index.size
    natural_positive_prob = total_positive_n / float(total_n)

    # Create Bins where First Bin has Observations with the
    # Highest Predicted Probability that y = 1
    df['BIN_POSITIVE'] = pd.qcut(df['PROB_POSITIVE'], bins, labels=False)

    pos_group_df = df.groupby('BIN_POSITIVE')
    # Percentage of Observations in each Bin where y = 1
    lift_positive = pos_group_df['ACTUAL'].sum() / pos_group_df['ACTUAL'].count()
    lift_index_positive = (lift_positive / natural_positive_prob) * 100

    # Consolidate Results into Output Dataframe
    lift_df = pd.DataFrame({'LIFT_POSITIVE': lift_positive,
                            'LIFT_POSITIVE_INDEX': lift_index_positive,
                            'BASELINE_POSITIVE': natural_positive_prob})
    print(lift_df)
    return lift_df
calc_lift(X, y, lr, bins=10)