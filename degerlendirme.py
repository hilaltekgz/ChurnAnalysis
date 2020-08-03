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

print('telcom',telcom)


#yes no değerlerinin sayılarını veriyor.
lab = telcom["Churn"].value_counts().keys().tolist()

val = telcom["Churn"].value_counts().values.tolist()
print('keys()',lab)
print('values',val)

# Aşağıdaki sütunlar için internet İnternet servisi yok’u Hayır olarak değiştirin.
replace_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies']
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


telcom["tenure_group"] = telcom.apply(lambda telcom: tenure_lab(telcom),
                                      axis=1)


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
print("num_cols",num_cols)
bin_cols = telcom.nunique()[telcom.nunique() == 2].keys().tolist()
print("bincols",bin_cols)
multi_cols = [i for i in cat_cols if i not in bin_cols]
print("multicols",multi_cols)



le = LabelEncoder()
for i in bin_cols:
    telcom[i] = le.fit_transform(telcom[i])


telcom = pd.get_dummies(data=telcom, columns=multi_cols)


std = StandardScaler()
scaled = abs(std.fit_transform(telcom[num_cols]))
print("scaled",len(scaled))
scaled = pd.DataFrame(scaled, columns=num_cols)


df_telcom_og = telcom.copy() #telcom copy
telcom = telcom.drop(columns=num_cols, axis=1)
print("telcominfooo",telcom.info())
telcom = telcom.merge(scaled, left_index=True, right_index=True, how="left")
print("telcommerge",telcom.info())
summary = (df_telcom_og[[i for i in df_telcom_og.columns if i not in Id_col]].
           describe(include='all').transpose().reset_index())
summary = summary.rename(columns = {"index" : "feature"})
summary = np.around(summary,3)
print(summary)

gl_obj = telcom.select_dtypes(include=['object']).copy()
aa = gl_obj.describe().T
print(aa)

gl_objb = telcom.select_dtypes(include=['int32']).copy()
bb = gl_objb.describe().T
print(bb)

gl_objc = telcom.select_dtypes(include=['uint8']).copy()
cc = gl_objc.describe().T
print(cc)

gl_objd = telcom.select_dtypes(include=['float64']).copy()
dd = gl_objd.describe()
print(dd)
attribute=[['gender','SeniorCitizen','Dependents','PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling',
            'MultipleLines_No','MultipleLines_No phone service','MultipleLines_Yes','InternetService_DSL','InternetService_Fiber optic','InternetService_No','Contract_Month-to-month','Contract_One year','Contract_Two year',
            'PaymentMethod_Electronic check','tenure_group_Tenure_0-24','tenure_group_Tenure_24-48','tenure_group_Tenure_48-72']]
for i in attribute:
    desc = telcom[i].describe().T
    print(desc)
descc = telcom['tenure_group_Tenure_48-72'].describe().T
print(descc)
# plt.scatter(telcom["tenure"], telcom["OnlineBackup"])
# plt.show()
# plt.scatter(telcom["tenure"], telcom["TotalCharges"])
# plt.show()
# plt.scatter(telcom["tenure"], telcom["gender"])
# plt.show()
# plt.scatter(telcom["gender"], telcom["tenure"])
# plt.show()
# plt.scatter(telcom["MonthlyCharges"], telcom["tenure"])
# plt.show()
# plt.scatter(telcom["MonthlyCharges"],telcom["TotalCharges"])
# plt.show()
# plt.scatter(telcom["StreamingTV"],telcom["TotalCharges"])
# plt.show()
#
# plt.scatter(telcom["OnlineSecurity"],telcom["OnlineBackup"])
# plt.show()
# print(telcom['tenure'])
# sns.pairplot(telcom,hue='Churn')
# plt.show()
print("telcominfo",telcom.info())
del telcom['customerID']
X= telcom.drop(columns='Churn', axis=1)
gender = pd.DataFrame(telcom['gender'])
print('gendertype',type(gender))
seniorCitizen = pd.DataFrame(telcom['SeniorCitizen'])
Dependents = pd.DataFrame(telcom['Dependents'])
PhoneService = pd.DataFrame(telcom['PhoneService'])
OnlineSecurity = pd.DataFrame(telcom['OnlineSecurity'])
OnlineBackup = pd.DataFrame(telcom['OnlineBackup'])
DeviceProtection = pd.DataFrame(telcom['DeviceProtection'])
TechSupport = pd.DataFrame(telcom['TechSupport'])
StreamingTV = pd.DataFrame(telcom['StreamingTV'])
StreamingMovies = pd.DataFrame(telcom['StreamingMovies'])
PaperlessBilling = pd.DataFrame(telcom['PaperlessBilling'])
MultipleLines_No = pd.DataFrame(telcom['MultipleLines_No'])
MultipleLines_No_phone_service = pd.DataFrame(telcom['MultipleLines_No phone service'])
MultipleLines_Yes = pd.DataFrame(telcom['MultipleLines_Yes'])
InternetService_DSL = pd.DataFrame(telcom['InternetService_DSL'])
InternetService_Fiber_optic = pd.DataFrame(telcom['InternetService_Fiber optic'])
InternetService_No = pd.DataFrame(telcom['InternetService_No'])
Contract_Month_to_month = pd.DataFrame(telcom['Contract_Month-to-month'])
Contract_One_year = pd.DataFrame(telcom['Contract_One year'])
Contract_Two_year = pd.DataFrame(telcom['Contract_Two year'])
PaymentMethod_Electronic_check = pd.DataFrame(telcom['PaymentMethod_Electronic check'])
PaymentMethod_Bank_transfer_automatic = pd.DataFrame(telcom['PaymentMethod_Bank transfer (automatic)'])
PaymentMethod_Credit_card_automatic = pd.DataFrame(telcom['PaymentMethod_Credit card (automatic)'])
PaymentMethod_Mailed_check = pd.DataFrame(telcom['PaymentMethod_Mailed check'])
tenure_group_Tenure_0_24 = pd.DataFrame(telcom['tenure_group_Tenure_0-24'])
tenure_group_Tenure_24_48 = pd.DataFrame(telcom['tenure_group_Tenure_24-48'])
tenure_group_Tenure_48_72 = pd.DataFrame(telcom['tenure_group_Tenure_48-72'])
tenure = pd.DataFrame(telcom['tenure'])
churn = pd.DataFrame(telcom['Churn'])
print(churn)
Partner = pd.DataFrame(telcom['Partner'])

data = pd.concat([gender,seniorCitizen,Partner,Dependents,churn] ,axis=1, join='inner')
sns.pairplot(data,hue='Churn',kind='reg')
plt.show()


data1 = pd.concat([gender,PhoneService,OnlineSecurity,OnlineBackup,DeviceProtection,churn] ,axis=1, join='inner')
sns.pairplot(data1,hue='Churn',kind='reg')
plt.show()


data2 = pd.concat([gender,TechSupport,StreamingTV,StreamingMovies,PaperlessBilling,MultipleLines_No,MultipleLines_No_phone_service,churn] ,axis=1, join='inner')
sns.pairplot(data2,hue='Churn',kind='reg')
plt.show()
data3 = pd.concat([gender,MultipleLines_Yes,InternetService_DSL,InternetService_Fiber_optic,InternetService_No,churn] ,axis=1, join='inner')
sns.pairplot(data3,hue='Churn',kind='reg')
plt.show()

data5 = pd.concat([gender,Contract_One_year,Contract_Two_year,PaymentMethod_Bank_transfer_automatic,churn] ,axis=1, join='inner')
sns.pairplot(data5,hue='Churn',kind='reg')
plt.show()

data6 = pd.concat([gender,PaymentMethod_Electronic_check,Contract_Month_to_month,PaymentMethod_Credit_card_automatic,churn] ,axis=1, join='inner')
sns.pairplot(data6,hue='Churn',kind='reg')
plt.show()

data4 = pd.concat([gender,PaymentMethod_Mailed_check,tenure_group_Tenure_0_24,tenure_group_Tenure_24_48,tenure_group_Tenure_48_72,tenure,churn] ,axis=1, join='inner')
sns.pairplot(data4,hue='Churn',kind='reg')
plt.show()
y = telcom['Churn']
from numpy import mean
from scipy.stats import pearsonr

print ('Pearsons correlation:gender, seniorCitizen ',np.corrcoef(telcom['gender'],telcom['SeniorCitizen']))
print('Pearsons correlation:gender, Partner  ',np.corrcoef(telcom['gender'],telcom['Partner']))
print('Pearsons correlation:gender, Dependents ',np.corrcoef(telcom['gender'],telcom['Dependents']))
print('Pearsons correlation:gender, PhoneService ',np.corrcoef(telcom['gender'],telcom['PhoneService']))
print('Pearsons correlation:gender, OnlineSecurity ',np.corrcoef(telcom['gender'],telcom['OnlineSecurity']))
print('Pearsons correlation:gender, OnlineBackup ',np.corrcoef(telcom['gender'],telcom['OnlineBackup']))
print('Pearsons correlation:gender,  DeviceProtection',np.corrcoef(telcom['gender'],telcom['DeviceProtection']))
print('Pearsons correlation:gender,  TechSupport',np.corrcoef(telcom['gender'],telcom['TechSupport']))
print('Pearsons correlation:gender, StreamingTV ',np.corrcoef(telcom['gender'],telcom['StreamingTV']))
print('Pearsons correlation:gender, StreamingMovies ',np.corrcoef(telcom['gender'],telcom['StreamingMovies']))
print('Pearsons correlation:gender,  PaperlessBilling',np.corrcoef(telcom['gender'],telcom['PaperlessBilling']))
print('Pearsons correlation:gender,  MultipleLines_No',np.corrcoef(telcom['gender'],telcom['MultipleLines_No']))
print('Pearsons correlation:gender,  MultipleLines_No phone service',np.corrcoef(telcom['gender'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:gender,  MultipleLines_Yes',np.corrcoef(telcom['gender'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:gender,  InternetService_DSL',np.corrcoef(telcom['gender'],telcom['InternetService_DSL']))
print('Pearsons correlation:gender,  InternetService_Fiber optic',np.corrcoef(telcom['gender'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:gender,  InternetService_No',np.corrcoef(telcom['gender'],telcom['InternetService_No']))
print('Pearsons correlation:gender, Contract_Month-to-month ',np.corrcoef(telcom['gender'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:gender,  Contract_One year',np.corrcoef(telcom['gender'],telcom['Contract_One year']))
print('Pearsons correlation:gender, Contract_Two year ',np.corrcoef(telcom['gender'],telcom['Contract_Two year']))
print('Pearsons correlation:gender,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['gender'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:gender,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['gender'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:gender,  PaymentMethod_Electronic check',np.corrcoef(telcom['gender'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:gender,  PaymentMethod_Electronic check',np.corrcoef(telcom['gender'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:gender, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['gender'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:gender, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['gender'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:gender, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['gender'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:gender, tenure ',np.corrcoef(telcom['gender'],telcom['tenure']))
print('Pearsons correlation:gender, MonthlyCharges ',np.corrcoef(telcom['gender'],telcom['MonthlyCharges']))
print('Pearsons correlation:gender, TotalCharges ',np.corrcoef(telcom['gender'],telcom['TotalCharges']))


print("-----------------------------------------------------------------------------------------------------")

print ('Pearsons correlation:seniorCitizen, gender ',np.corrcoef(telcom['SeniorCitizen'],telcom['gender']))
print('Pearsons correlation:SeniorCitizen, Partner  ',np.corrcoef(telcom['SeniorCitizen'],telcom['Partner']))
print('Pearsons correlation:SeniorCitizen, Dependents ',np.corrcoef(telcom['SeniorCitizen'],telcom['Dependents']))
print('Pearsons correlation:SeniorCitizen, PhoneService ',np.corrcoef(telcom['SeniorCitizen'],telcom['PhoneService']))
print('Pearsons correlation:SeniorCitizen, OnlineSecurity ',np.corrcoef(telcom['SeniorCitizen'],telcom['OnlineSecurity']))
print('Pearsons correlation:SeniorCitizen, OnlineBackup ',np.corrcoef(telcom['SeniorCitizen'],telcom['OnlineBackup']))
print('Pearsons correlation:SeniorCitizen,  DeviceProtection',np.corrcoef(telcom['SeniorCitizen'],telcom['DeviceProtection']))
print('Pearsons correlation:SeniorCitizen,  TechSupport',np.corrcoef(telcom['SeniorCitizen'],telcom['TechSupport']))
print('Pearsons correlation:SeniorCitizen, StreamingTV ',np.corrcoef(telcom['SeniorCitizen'],telcom['StreamingTV']))
print('Pearsons correlation:SeniorCitizen, StreamingMovies ',np.corrcoef(telcom['SeniorCitizen'],telcom['StreamingMovies']))
print('Pearsons correlation:SeniorCitizen,  PaperlessBilling',np.corrcoef(telcom['SeniorCitizen'],telcom['PaperlessBilling']))
print('Pearsons correlation:SeniorCitizen,  MultipleLines_No',np.corrcoef(telcom['SeniorCitizen'],telcom['MultipleLines_No']))
print('Pearsons correlation:SeniorCitizen,  MultipleLines_No phone service',np.corrcoef(telcom['SeniorCitizen'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:SeniorCitizen,  MultipleLines_Yes',np.corrcoef(telcom['SeniorCitizen'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:SeniorCitizen,  InternetService_DSL',np.corrcoef(telcom['SeniorCitizen'],telcom['InternetService_DSL']))
print('Pearsons correlation:SeniorCitizen,  InternetService_Fiber optic',np.corrcoef(telcom['SeniorCitizen'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:SeniorCitizen,  InternetService_No',np.corrcoef(telcom['SeniorCitizen'],telcom['InternetService_No']))
print('Pearsons correlation:SeniorCitizen, Contract_Month-to-month ',np.corrcoef(telcom['SeniorCitizen'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:SeniorCitizen,  Contract_One year',np.corrcoef(telcom['SeniorCitizen'],telcom['Contract_One year']))
print('Pearsons correlation:SeniorCitizen, Contract_Two year ',np.corrcoef(telcom['SeniorCitizen'],telcom['Contract_Two year']))
print('Pearsons correlation:SeniorCitizen,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['SeniorCitizen'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:SeniorCitizen,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['SeniorCitizen'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:SeniorCitizen,  PaymentMethod_Electronic check',np.corrcoef(telcom['SeniorCitizen'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:SeniorCitizen,  PaymentMethod_Electronic check',np.corrcoef(telcom['SeniorCitizen'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:SeniorCitizen, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['SeniorCitizen'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:SeniorCitizen, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['SeniorCitizen'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:SeniorCitizen, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['SeniorCitizen'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:SeniorCitizen, tenure ',np.corrcoef(telcom['SeniorCitizen'],telcom['tenure']))
print('Pearsons correlation:SeniorCitizen, MonthlyCharges ',np.corrcoef(telcom['SeniorCitizen'],telcom['MonthlyCharges']))
print('Pearsons correlation:SeniorCitizen, TotalCharges ',np.corrcoef(telcom['SeniorCitizen'],telcom['TotalCharges']))



print('--------------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:Dependents, gender ',np.corrcoef(telcom['Dependents'],telcom['gender']))
print('Pearsons correlation:Dependents, Partner  ',np.corrcoef(telcom['Dependents'],telcom['Partner']))
print('Pearsons correlation:Dependents, PhoneService ',np.corrcoef(telcom['Dependents'],telcom['PhoneService']))
print('Pearsons correlation:Dependents, OnlineSecurity ',np.corrcoef(telcom['Dependents'],telcom['OnlineSecurity']))
print('Pearsons correlation:Dependents, OnlineBackup ',np.corrcoef(telcom['Dependents'],telcom['OnlineBackup']))
print('Pearsons correlation:Dependents,  DeviceProtection',np.corrcoef(telcom['Dependents'],telcom['DeviceProtection']))
print('Pearsons correlation:Dependents,  TechSupport',np.corrcoef(telcom['Dependents'],telcom['TechSupport']))
print('Pearsons correlation:Dependents, StreamingTV ',np.corrcoef(telcom['Dependents'],telcom['StreamingTV']))
print('Pearsons correlation:Dependents, StreamingMovies ',np.corrcoef(telcom['Dependents'],telcom['StreamingMovies']))
print('Pearsons correlation:Dependents,  PaperlessBilling',np.corrcoef(telcom['Dependents'],telcom['PaperlessBilling']))
print('Pearsons correlation:Dependents,  MultipleLines_No',np.corrcoef(telcom['Dependents'],telcom['MultipleLines_No']))
print('Pearsons correlation:Dependents,  MultipleLines_No phone service',np.corrcoef(telcom['Dependents'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:Dependents,  MultipleLines_Yes',np.corrcoef(telcom['Dependents'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:Dependents,  InternetService_DSL',np.corrcoef(telcom['Dependents'],telcom['InternetService_DSL']))
print('Pearsons correlation:Dependents,  InternetService_Fiber optic',np.corrcoef(telcom['Dependents'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:Dependents,  InternetService_No',np.corrcoef(telcom['Dependents'],telcom['InternetService_No']))
print('Pearsons correlation:Dependents, Contract_Month-to-month ',np.corrcoef(telcom['Dependents'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:Dependents,  Contract_One year',np.corrcoef(telcom['Dependents'],telcom['Contract_One year']))
print('Pearsons correlation:Dependents, Contract_Two year ',np.corrcoef(telcom['Dependents'],telcom['Contract_Two year']))
print('Pearsons correlation:Dependents,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['Dependents'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:Dependents,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['Dependents'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:Dependents,  PaymentMethod_Electronic check',np.corrcoef(telcom['Dependents'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:Dependents,  PaymentMethod_Electronic check',np.corrcoef(telcom['Dependents'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:Dependents, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['Dependents'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:Dependents, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['Dependents'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:Dependents, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['Dependents'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:Dependents, tenure ',np.corrcoef(telcom['Dependents'],telcom['tenure']))
print('Pearsons correlation:Dependents, MonthlyCharges ',np.corrcoef(telcom['Dependents'],telcom['MonthlyCharges']))
print('Pearsons correlation:Dependents, TotalCharges ',np.corrcoef(telcom['Dependents'],telcom['TotalCharges']))

print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:PhoneService, gender ',np.corrcoef(telcom['PhoneService'],telcom['gender']))
print('Pearsons correlation:PhoneService, Partner  ',np.corrcoef(telcom['PhoneService'],telcom['Partner']))
print('Pearsons correlation:PhoneService, PhoneService ',np.corrcoef(telcom['PhoneService'],telcom['PhoneService']))
print('Pearsons correlation:PhoneService, OnlineSecurity ',np.corrcoef(telcom['PhoneService'],telcom['OnlineSecurity']))
print('Pearsons correlation:PhoneService, OnlineBackup ',np.corrcoef(telcom['PhoneService'],telcom['OnlineBackup']))
print('Pearsons correlation:PhoneService,  DeviceProtection',np.corrcoef(telcom['PhoneService'],telcom['DeviceProtection']))
print('Pearsons correlation:PhoneService,  TechSupport',np.corrcoef(telcom['PhoneService'],telcom['TechSupport']))
print('Pearsons correlation:PhoneService, StreamingTV ',np.corrcoef(telcom['PhoneService'],telcom['StreamingTV']))
print('Pearsons correlation:PhoneService, StreamingMovies ',np.corrcoef(telcom['PhoneService'],telcom['StreamingMovies']))
print('Pearsons correlation:PhoneService,  PaperlessBilling',np.corrcoef(telcom['PhoneService'],telcom['PaperlessBilling']))
print('Pearsons correlation:PhoneService,  MultipleLines_No',np.corrcoef(telcom['PhoneService'],telcom['MultipleLines_No']))
print('Pearsons correlation:PhoneService,  MultipleLines_No phone service',np.corrcoef(telcom['PhoneService'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:PhoneService,  MultipleLines_Yes',np.corrcoef(telcom['PhoneService'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:PhoneService,  InternetService_DSL',np.corrcoef(telcom['PhoneService'],telcom['InternetService_DSL']))
print('Pearsons correlation:PhoneService,  InternetService_Fiber optic',np.corrcoef(telcom['PhoneService'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:PhoneService,  InternetService_No',np.corrcoef(telcom['PhoneService'],telcom['InternetService_No']))
print('Pearsons correlation:PhoneService, Contract_Month-to-month ',np.corrcoef(telcom['PhoneService'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:PhoneService,  Contract_One year',np.corrcoef(telcom['PhoneService'],telcom['Contract_One year']))
print('Pearsons correlation:PhoneService, Contract_Two year ',np.corrcoef(telcom['PhoneService'],telcom['Contract_Two year']))
print('Pearsons correlation:PhoneService,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['PhoneService'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:PhoneService,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['PhoneService'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:PhoneService,  PaymentMethod_Electronic check',np.corrcoef(telcom['PhoneService'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:PhoneService,  PaymentMethod_Electronic check',np.corrcoef(telcom['PhoneService'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:PhoneService, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['PhoneService'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:PhoneService, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['PhoneService'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:PhoneService, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['PhoneService'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:PhoneService, tenure ',np.corrcoef(telcom['PhoneService'],telcom['tenure']))
print('Pearsons correlation:PhoneService, MonthlyCharges ',np.corrcoef(telcom['PhoneService'],telcom['MonthlyCharges']))
print('Pearsons correlation:PhoneService, TotalCharges ',np.corrcoef(telcom['PhoneService'],telcom['TotalCharges']))

print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:OnlineSecurity, gender ',np.corrcoef(telcom['OnlineSecurity'],telcom['gender']))
print('Pearsons correlation:OnlineSecurity, Partner  ',np.corrcoef(telcom['OnlineSecurity'],telcom['Partner']))
print('Pearsons correlation:OnlineSecurity, PhoneService ',np.corrcoef(telcom['OnlineSecurity'],telcom['PhoneService']))
print('Pearsons correlation:OnlineSecurity, OnlineSecurity ',np.corrcoef(telcom['OnlineSecurity'],telcom['OnlineSecurity']))
print('Pearsons correlation:OnlineSecurity, OnlineBackup ',np.corrcoef(telcom['OnlineSecurity'],telcom['OnlineBackup']))
print('Pearsons correlation:OnlineSecurity,  DeviceProtection',np.corrcoef(telcom['OnlineSecurity'],telcom['DeviceProtection']))
print('Pearsons correlation:OnlineSecurity,  TechSupport',np.corrcoef(telcom['OnlineSecurity'],telcom['TechSupport']))
print('Pearsons correlation:OnlineSecurity, StreamingTV ',np.corrcoef(telcom['OnlineSecurity'],telcom['StreamingTV']))
print('Pearsons correlation:OnlineSecurity, StreamingMovies ',np.corrcoef(telcom['OnlineSecurity'],telcom['StreamingMovies']))
print('Pearsons correlation:OnlineSecurity,  PaperlessBilling',np.corrcoef(telcom['OnlineSecurity'],telcom['PaperlessBilling']))
print('Pearsons correlation:OnlineSecurity,  MultipleLines_No',np.corrcoef(telcom['OnlineSecurity'],telcom['MultipleLines_No']))
print('Pearsons correlation:OnlineSecurity,  MultipleLines_No phone service',np.corrcoef(telcom['OnlineSecurity'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:OnlineSecurity,  MultipleLines_Yes',np.corrcoef(telcom['OnlineSecurity'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:OnlineSecurity,  InternetService_DSL',np.corrcoef(telcom['OnlineSecurity'],telcom['InternetService_DSL']))
print('Pearsons correlation:OnlineSecurity,  InternetService_Fiber optic',np.corrcoef(telcom['OnlineSecurity'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:OnlineSecurity,  InternetService_No',np.corrcoef(telcom['OnlineSecurity'],telcom['InternetService_No']))
print('Pearsons correlation:OnlineSecurity, Contract_Month-to-month ',np.corrcoef(telcom['OnlineSecurity'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:OnlineSecurity,  Contract_One year',np.corrcoef(telcom['OnlineSecurity'],telcom['Contract_One year']))
print('Pearsons correlation:OnlineSecurity, Contract_Two year ',np.corrcoef(telcom['OnlineSecurity'],telcom['Contract_Two year']))
print('Pearsons correlation:OnlineSecurity,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['OnlineSecurity'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:OnlineSecurity,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['OnlineSecurity'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:OnlineSecurity,  PaymentMethod_Electronic check',np.corrcoef(telcom['OnlineSecurity'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:OnlineSecurity,  PaymentMethod_Electronic check',np.corrcoef(telcom['OnlineSecurity'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:OnlineSecurity, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['OnlineSecurity'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:OnlineSecurity, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['OnlineSecurity'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:OnlineSecurity, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['OnlineSecurity'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:OnlineSecurity, tenure ',np.corrcoef(telcom['OnlineSecurity'],telcom['tenure']))
print('Pearsons correlation:OnlineSecurity, MonthlyCharges ',np.corrcoef(telcom['OnlineSecurity'],telcom['MonthlyCharges']))
print('Pearsons correlation:OnlineSecurity, TotalCharges ',np.corrcoef(telcom['OnlineSecurity'],telcom['TotalCharges']))


print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:OnlineBackup, gender ',np.corrcoef(telcom['OnlineBackup'],telcom['gender']))
print('Pearsons correlation:OnlineBackup, Partner  ',np.corrcoef(telcom['OnlineBackup'],telcom['Partner']))
print('Pearsons correlation:OnlineBackup, PhoneService ',np.corrcoef(telcom['OnlineBackup'],telcom['PhoneService']))
print('Pearsons correlation:OnlineBackup, OnlineSecurity ',np.corrcoef(telcom['OnlineBackup'],telcom['OnlineSecurity']))
print('Pearsons correlation:OnlineBackup, OnlineBackup ',np.corrcoef(telcom['OnlineBackup'],telcom['OnlineBackup']))
print('Pearsons correlation:OnlineBackup,  DeviceProtection',np.corrcoef(telcom['OnlineBackup'],telcom['DeviceProtection']))
print('Pearsons correlation:OnlineBackup,  TechSupport',np.corrcoef(telcom['OnlineBackup'],telcom['TechSupport']))
print('Pearsons correlation:OnlineBackup, StreamingTV ',np.corrcoef(telcom['OnlineBackup'],telcom['StreamingTV']))
print('Pearsons correlation:OnlineBackup, StreamingMovies ',np.corrcoef(telcom['OnlineBackup'],telcom['StreamingMovies']))
print('Pearsons correlation:OnlineBackup,  PaperlessBilling',np.corrcoef(telcom['OnlineBackup'],telcom['PaperlessBilling']))
print('Pearsons correlation:OnlineBackup,  MultipleLines_No',np.corrcoef(telcom['OnlineBackup'],telcom['MultipleLines_No']))
print('Pearsons correlation:OnlineBackup,  MultipleLines_No phone service',np.corrcoef(telcom['OnlineBackup'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:OnlineBackup,  MultipleLines_Yes',np.corrcoef(telcom['OnlineBackup'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:OnlineBackup,  InternetService_DSL',np.corrcoef(telcom['OnlineBackup'],telcom['InternetService_DSL']))
print('Pearsons correlation:OnlineBackup,  InternetService_Fiber optic',np.corrcoef(telcom['OnlineBackup'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:OnlineBackup,  InternetService_No',np.corrcoef(telcom['OnlineBackup'],telcom['InternetService_No']))
print('Pearsons correlation:OnlineBackup, Contract_Month-to-month ',np.corrcoef(telcom['OnlineBackup'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:OnlineBackup,  Contract_One year',np.corrcoef(telcom['OnlineBackup'],telcom['Contract_One year']))
print('Pearsons correlation:OnlineBackup, Contract_Two year ',np.corrcoef(telcom['OnlineBackup'],telcom['Contract_Two year']))
print('Pearsons correlation:OnlineBackup,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['OnlineBackup'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:OnlineBackup,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['OnlineBackup'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:OnlineBackup,  PaymentMethod_Electronic check',np.corrcoef(telcom['OnlineBackup'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:OnlineBackup,  PaymentMethod_Electronic check',np.corrcoef(telcom['OnlineBackup'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:OnlineBackup, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['OnlineBackup'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:OnlineBackup, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['OnlineBackup'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:OnlineBackup, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['OnlineBackup'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:OnlineBackup, tenure ',np.corrcoef(telcom['OnlineBackup'],telcom['tenure']))
print('Pearsons correlation:OnlineBackup, MonthlyCharges ',np.corrcoef(telcom['OnlineBackup'],telcom['MonthlyCharges']))
print('Pearsons correlation:OnlineBackup, TotalCharges ',np.corrcoef(telcom['OnlineBackup'],telcom['TotalCharges']))


print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:DeviceProtection, gender ',np.corrcoef(telcom['DeviceProtection'],telcom['gender']))
print('Pearsons correlation:DeviceProtection, Partner  ',np.corrcoef(telcom['DeviceProtection'],telcom['Partner']))
print('Pearsons correlation:DeviceProtection, PhoneService ',np.corrcoef(telcom['DeviceProtection'],telcom['PhoneService']))
print('Pearsons correlation:DeviceProtection, OnlineSecurity ',np.corrcoef(telcom['DeviceProtection'],telcom['OnlineSecurity']))
print('Pearsons correlation:DeviceProtection, OnlineBackup ',np.corrcoef(telcom['DeviceProtection'],telcom['OnlineBackup']))
print('Pearsons correlation:DeviceProtection,  DeviceProtection',np.corrcoef(telcom['DeviceProtection'],telcom['DeviceProtection']))
print('Pearsons correlation:DeviceProtection,  TechSupport',np.corrcoef(telcom['DeviceProtection'],telcom['TechSupport']))
print('Pearsons correlation:DeviceProtection, StreamingTV ',np.corrcoef(telcom['DeviceProtection'],telcom['StreamingTV']))
print('Pearsons correlation:DeviceProtection, StreamingMovies ',np.corrcoef(telcom['DeviceProtection'],telcom['StreamingMovies']))
print('Pearsons correlation:DeviceProtection,  PaperlessBilling',np.corrcoef(telcom['DeviceProtection'],telcom['PaperlessBilling']))
print('Pearsons correlation:DeviceProtection,  MultipleLines_No',np.corrcoef(telcom['DeviceProtection'],telcom['MultipleLines_No']))
print('Pearsons correlation:DeviceProtection,  MultipleLines_No phone service',np.corrcoef(telcom['DeviceProtection'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:DeviceProtection,  MultipleLines_Yes',np.corrcoef(telcom['DeviceProtection'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:DeviceProtection,  InternetService_DSL',np.corrcoef(telcom['DeviceProtection'],telcom['InternetService_DSL']))
print('Pearsons correlation:DeviceProtection,  InternetService_Fiber optic',np.corrcoef(telcom['DeviceProtection'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:DeviceProtection,  InternetService_No',np.corrcoef(telcom['DeviceProtection'],telcom['InternetService_No']))
print('Pearsons correlation:DeviceProtection, Contract_Month-to-month ',np.corrcoef(telcom['DeviceProtection'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:DeviceProtection,  Contract_One year',np.corrcoef(telcom['DeviceProtection'],telcom['Contract_One year']))
print('Pearsons correlation:DeviceProtection, Contract_Two year ',np.corrcoef(telcom['DeviceProtection'],telcom['Contract_Two year']))
print('Pearsons correlation:DeviceProtection,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['DeviceProtection'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:DeviceProtection,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['DeviceProtection'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:DeviceProtection,  PaymentMethod_Electronic check',np.corrcoef(telcom['DeviceProtection'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:DeviceProtection,  PaymentMethod_Electronic check',np.corrcoef(telcom['DeviceProtection'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:DeviceProtection, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['DeviceProtection'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:DeviceProtection, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['DeviceProtection'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:DeviceProtection, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['DeviceProtection'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:DeviceProtection, tenure ',np.corrcoef(telcom['DeviceProtection'],telcom['tenure']))
print('Pearsons correlation:DeviceProtection, MonthlyCharges ',np.corrcoef(telcom['DeviceProtection'],telcom['MonthlyCharges']))
print('Pearsons correlation:DeviceProtection, TotalCharges ',np.corrcoef(telcom['DeviceProtection'],telcom['TotalCharges']))

print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:TechSupport, gender ',np.corrcoef(telcom['TechSupport'],telcom['gender']))
print('Pearsons correlation:TechSupport, Partner  ',np.corrcoef(telcom['TechSupport'],telcom['Partner']))
print('Pearsons correlation:TechSupport, PhoneService ',np.corrcoef(telcom['TechSupport'],telcom['PhoneService']))
print('Pearsons correlation:TechSupport, OnlineSecurity ',np.corrcoef(telcom['TechSupport'],telcom['OnlineSecurity']))
print('Pearsons correlation:TechSupport, OnlineBackup ',np.corrcoef(telcom['TechSupport'],telcom['OnlineBackup']))
print('Pearsons correlation:TechSupport,  DeviceProtection',np.corrcoef(telcom['TechSupport'],telcom['DeviceProtection']))
print('Pearsons correlation:TechSupport,  TechSupport',np.corrcoef(telcom['TechSupport'],telcom['TechSupport']))
print('Pearsons correlation:TechSupport, StreamingTV ',np.corrcoef(telcom['TechSupport'],telcom['StreamingTV']))
print('Pearsons correlation:TechSupport, StreamingMovies ',np.corrcoef(telcom['TechSupport'],telcom['StreamingMovies']))
print('Pearsons correlation:TechSupport,  PaperlessBilling',np.corrcoef(telcom['TechSupport'],telcom['PaperlessBilling']))
print('Pearsons correlation:TechSupport,  MultipleLines_No',np.corrcoef(telcom['TechSupport'],telcom['MultipleLines_No']))
print('Pearsons correlation:TechSupport,  MultipleLines_No phone service',np.corrcoef(telcom['TechSupport'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:TechSupport,  MultipleLines_Yes',np.corrcoef(telcom['TechSupport'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:TechSupport,  InternetService_DSL',np.corrcoef(telcom['TechSupport'],telcom['InternetService_DSL']))
print('Pearsons correlation:TechSupport,  InternetService_Fiber optic',np.corrcoef(telcom['TechSupport'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:TechSupport,  InternetService_No',np.corrcoef(telcom['TechSupport'],telcom['InternetService_No']))
print('Pearsons correlation:TechSupport, Contract_Month-to-month ',np.corrcoef(telcom['TechSupport'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:TechSupport,  Contract_One year',np.corrcoef(telcom['TechSupport'],telcom['Contract_One year']))
print('Pearsons correlation:TechSupport, Contract_Two year ',np.corrcoef(telcom['TechSupport'],telcom['Contract_Two year']))
print('Pearsons correlation:TechSupport,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['TechSupport'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:TechSupport,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['TechSupport'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:TechSupport,  PaymentMethod_Electronic check',np.corrcoef(telcom['TechSupport'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:TechSupport,  PaymentMethod_Electronic check',np.corrcoef(telcom['TechSupport'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:TechSupport, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['TechSupport'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:TechSupport, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['TechSupport'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:TechSupport, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['TechSupport'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:TechSupport, tenure ',np.corrcoef(telcom['TechSupport'],telcom['tenure']))
print('Pearsons correlation:TechSupport, MonthlyCharges ',np.corrcoef(telcom['TechSupport'],telcom['MonthlyCharges']))
print('Pearsons correlation:TechSupport, TotalCharges ',np.corrcoef(telcom['TechSupport'],telcom['TotalCharges']))

print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:StreamingTV, gender ',np.corrcoef(telcom['StreamingTV'],telcom['gender']))
print('Pearsons correlation:StreamingTV, Partner  ',np.corrcoef(telcom['StreamingTV'],telcom['Partner']))
print('Pearsons correlation:StreamingTV, PhoneService ',np.corrcoef(telcom['StreamingTV'],telcom['PhoneService']))
print('Pearsons correlation:StreamingTV, OnlineSecurity ',np.corrcoef(telcom['StreamingTV'],telcom['OnlineSecurity']))
print('Pearsons correlation:StreamingTV, OnlineBackup ',np.corrcoef(telcom['StreamingTV'],telcom['OnlineBackup']))
print('Pearsons correlation:StreamingTV,  DeviceProtection',np.corrcoef(telcom['StreamingTV'],telcom['DeviceProtection']))
print('Pearsons correlation:StreamingTV,  TechSupport',np.corrcoef(telcom['StreamingTV'],telcom['TechSupport']))
print('Pearsons correlation:StreamingTV, StreamingTV ',np.corrcoef(telcom['StreamingTV'],telcom['StreamingTV']))
print('Pearsons correlation:StreamingTV, StreamingMovies ',np.corrcoef(telcom['StreamingTV'],telcom['StreamingMovies']))
print('Pearsons correlation:StreamingTV,  PaperlessBilling',np.corrcoef(telcom['StreamingTV'],telcom['PaperlessBilling']))
print('Pearsons correlation:StreamingTV,  MultipleLines_No',np.corrcoef(telcom['StreamingTV'],telcom['MultipleLines_No']))
print('Pearsons correlation:StreamingTV,  MultipleLines_No phone service',np.corrcoef(telcom['StreamingTV'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:StreamingTV,  MultipleLines_Yes',np.corrcoef(telcom['StreamingTV'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:StreamingTV,  InternetService_DSL',np.corrcoef(telcom['StreamingTV'],telcom['InternetService_DSL']))
print('Pearsons correlation:StreamingTV,  InternetService_Fiber optic',np.corrcoef(telcom['StreamingTV'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:StreamingTV,  InternetService_No',np.corrcoef(telcom['StreamingTV'],telcom['InternetService_No']))
print('Pearsons correlation:StreamingTV, Contract_Month-to-month ',np.corrcoef(telcom['StreamingTV'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:StreamingTV,  Contract_One year',np.corrcoef(telcom['StreamingTV'],telcom['Contract_One year']))
print('Pearsons correlation:StreamingTV, Contract_Two year ',np.corrcoef(telcom['StreamingTV'],telcom['Contract_Two year']))
print('Pearsons correlation:StreamingTV,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['StreamingTV'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:StreamingTV,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['StreamingTV'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:StreamingTV,  PaymentMethod_Electronic check',np.corrcoef(telcom['StreamingTV'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:StreamingTV,  PaymentMethod_Electronic check',np.corrcoef(telcom['StreamingTV'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:StreamingTV, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['StreamingTV'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:StreamingTV, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['StreamingTV'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:StreamingTV, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['StreamingTV'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:StreamingTV, tenure ',np.corrcoef(telcom['StreamingTV'],telcom['tenure']))
print('Pearsons correlation:StreamingTV, MonthlyCharges ',np.corrcoef(telcom['StreamingTV'],telcom['MonthlyCharges']))
print('Pearsons correlation:StreamingTV, TotalCharges ',np.corrcoef(telcom['StreamingTV'],telcom['TotalCharges']))


print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:StreamingMovies, gender ',np.corrcoef(telcom['StreamingMovies'],telcom['gender']))
print('Pearsons correlation:StreamingMovies, Partner  ',np.corrcoef(telcom['StreamingMovies'],telcom['Partner']))
print('Pearsons correlation:StreamingMovies, PhoneService ',np.corrcoef(telcom['StreamingMovies'],telcom['PhoneService']))
print('Pearsons correlation:StreamingMovies, OnlineSecurity ',np.corrcoef(telcom['StreamingMovies'],telcom['OnlineSecurity']))
print('Pearsons correlation:StreamingMovies, OnlineBackup ',np.corrcoef(telcom['StreamingMovies'],telcom['OnlineBackup']))
print('Pearsons correlation:StreamingMovies,  DeviceProtection',np.corrcoef(telcom['StreamingMovies'],telcom['DeviceProtection']))
print('Pearsons correlation:StreamingMovies,  TechSupport',np.corrcoef(telcom['StreamingMovies'],telcom['TechSupport']))
print('Pearsons correlation:StreamingMovies, StreamingTV ',np.corrcoef(telcom['StreamingMovies'],telcom['StreamingTV']))
print('Pearsons correlation:StreamingMovies, StreamingMovies ',np.corrcoef(telcom['StreamingMovies'],telcom['StreamingMovies']))
print('Pearsons correlation:StreamingMovies,  PaperlessBilling',np.corrcoef(telcom['StreamingMovies'],telcom['PaperlessBilling']))
print('Pearsons correlation:StreamingMovies,  MultipleLines_No',np.corrcoef(telcom['StreamingMovies'],telcom['MultipleLines_No']))
print('Pearsons correlation:StreamingMovies,  MultipleLines_No phone service',np.corrcoef(telcom['StreamingMovies'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:StreamingMovies,  MultipleLines_Yes',np.corrcoef(telcom['StreamingMovies'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:StreamingMovies,  InternetService_DSL',np.corrcoef(telcom['StreamingMovies'],telcom['InternetService_DSL']))
print('Pearsons correlation:StreamingMovies,  InternetService_Fiber optic',np.corrcoef(telcom['StreamingMovies'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:StreamingMovies,  InternetService_No',np.corrcoef(telcom['StreamingMovies'],telcom['InternetService_No']))
print('Pearsons correlation:StreamingMovies, Contract_Month-to-month ',np.corrcoef(telcom['StreamingMovies'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:StreamingMovies,  Contract_One year',np.corrcoef(telcom['StreamingMovies'],telcom['Contract_One year']))
print('Pearsons correlation:StreamingMovies, Contract_Two year ',np.corrcoef(telcom['StreamingMovies'],telcom['Contract_Two year']))
print('Pearsons correlation:StreamingMovies,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['StreamingMovies'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:StreamingMovies,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['StreamingMovies'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:StreamingMovies,  PaymentMethod_Electronic check',np.corrcoef(telcom['StreamingMovies'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:StreamingMovies,  PaymentMethod_Electronic check',np.corrcoef(telcom['StreamingMovies'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:StreamingMovies, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['StreamingMovies'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:StreamingMovies, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['StreamingMovies'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:StreamingMovies, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['StreamingMovies'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:StreamingMovies, tenure ',np.corrcoef(telcom['StreamingMovies'],telcom['tenure']))
print('Pearsons correlation:StreamingMovies, MonthlyCharges ',np.corrcoef(telcom['StreamingMovies'],telcom['MonthlyCharges']))
print('Pearsons correlation:StreamingMovies, TotalCharges ',np.corrcoef(telcom['StreamingMovies'],telcom['TotalCharges']))



print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:PaperlessBilling, gender ',np.corrcoef(telcom['PaperlessBilling'],telcom['gender']))
print('Pearsons correlation:PaperlessBilling, Partner  ',np.corrcoef(telcom['PaperlessBilling'],telcom['Partner']))
print('Pearsons correlation:PaperlessBilling, PhoneService ',np.corrcoef(telcom['PaperlessBilling'],telcom['PhoneService']))
print('Pearsons correlation:PaperlessBilling, OnlineSecurity ',np.corrcoef(telcom['PaperlessBilling'],telcom['OnlineSecurity']))
print('Pearsons correlation:PaperlessBilling, OnlineBackup ',np.corrcoef(telcom['PaperlessBilling'],telcom['OnlineBackup']))
print('Pearsons correlation:PaperlessBilling,  DeviceProtection',np.corrcoef(telcom['PaperlessBilling'],telcom['DeviceProtection']))
print('Pearsons correlation:PaperlessBilling,  TechSupport',np.corrcoef(telcom['PaperlessBilling'],telcom['TechSupport']))
print('Pearsons correlation:PaperlessBilling, StreamingTV ',np.corrcoef(telcom['PaperlessBilling'],telcom['StreamingTV']))
print('Pearsons correlation:PaperlessBilling, StreamingMovies ',np.corrcoef(telcom['PaperlessBilling'],telcom['StreamingMovies']))
print('Pearsons correlation:PaperlessBilling,  PaperlessBilling',np.corrcoef(telcom['PaperlessBilling'],telcom['PaperlessBilling']))
print('Pearsons correlation:PaperlessBilling,  MultipleLines_No',np.corrcoef(telcom['PaperlessBilling'],telcom['MultipleLines_No']))
print('Pearsons correlation:PaperlessBilling,  MultipleLines_No phone service',np.corrcoef(telcom['PaperlessBilling'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:PaperlessBilling,  MultipleLines_Yes',np.corrcoef(telcom['PaperlessBilling'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:PaperlessBilling,  InternetService_DSL',np.corrcoef(telcom['PaperlessBilling'],telcom['InternetService_DSL']))
print('Pearsons correlation:PaperlessBilling,  InternetService_Fiber optic',np.corrcoef(telcom['PaperlessBilling'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:PaperlessBilling,  InternetService_No',np.corrcoef(telcom['PaperlessBilling'],telcom['InternetService_No']))
print('Pearsons correlation:PaperlessBilling, Contract_Month-to-month ',np.corrcoef(telcom['PaperlessBilling'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:PaperlessBilling,  Contract_One year',np.corrcoef(telcom['PaperlessBilling'],telcom['Contract_One year']))
print('Pearsons correlation:PaperlessBilling, Contract_Two year ',np.corrcoef(telcom['PaperlessBilling'],telcom['Contract_Two year']))
print('Pearsons correlation:PaperlessBilling,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['PaperlessBilling'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:PaperlessBilling,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['PaperlessBilling'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:PaperlessBilling,  PaymentMethod_Electronic check',np.corrcoef(telcom['PaperlessBilling'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:PaperlessBilling,  PaymentMethod_Electronic check',np.corrcoef(telcom['PaperlessBilling'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:PaperlessBilling, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['PaperlessBilling'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:PaperlessBilling, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['PaperlessBilling'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:PaperlessBilling, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['PaperlessBilling'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:PaperlessBilling, tenure ',np.corrcoef(telcom['PaperlessBilling'],telcom['tenure']))
print('Pearsons correlation:PaperlessBilling, MonthlyCharges ',np.corrcoef(telcom['PaperlessBilling'],telcom['MonthlyCharges']))
print('Pearsons correlation:PaperlessBilling, TotalCharges ',np.corrcoef(telcom['PaperlessBilling'],telcom['TotalCharges']))



print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:MultipleLines_No, gender ',np.corrcoef(telcom['MultipleLines_No'],telcom['gender']))
print('Pearsons correlation:MultipleLines_No, Partner  ',np.corrcoef(telcom['MultipleLines_No'],telcom['Partner']))
print('Pearsons correlation:MultipleLines_No, PhoneService ',np.corrcoef(telcom['MultipleLines_No'],telcom['PhoneService']))
print('Pearsons correlation:MultipleLines_No, OnlineSecurity ',np.corrcoef(telcom['MultipleLines_No'],telcom['OnlineSecurity']))
print('Pearsons correlation:MultipleLines_No, OnlineBackup ',np.corrcoef(telcom['MultipleLines_No'],telcom['OnlineBackup']))
print('Pearsons correlation:MultipleLines_No,  DeviceProtection',np.corrcoef(telcom['MultipleLines_No'],telcom['DeviceProtection']))
print('Pearsons correlation:MultipleLines_No,  TechSupport',np.corrcoef(telcom['MultipleLines_No'],telcom['TechSupport']))
print('Pearsons correlation:MultipleLines_No, StreamingTV ',np.corrcoef(telcom['MultipleLines_No'],telcom['StreamingTV']))
print('Pearsons correlation:MultipleLines_No, StreamingMovies ',np.corrcoef(telcom['MultipleLines_No'],telcom['StreamingMovies']))
print('Pearsons correlation:MultipleLines_No,  PaperlessBilling',np.corrcoef(telcom['MultipleLines_No'],telcom['PaperlessBilling']))
print('Pearsons correlation:MultipleLines_No,  MultipleLines_No',np.corrcoef(telcom['MultipleLines_No'],telcom['MultipleLines_No']))
print('Pearsons correlation:MultipleLines_No,  MultipleLines_No phone service',np.corrcoef(telcom['MultipleLines_No'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:MultipleLines_No,  MultipleLines_Yes',np.corrcoef(telcom['MultipleLines_No'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:MultipleLines_No,  InternetService_DSL',np.corrcoef(telcom['MultipleLines_No'],telcom['InternetService_DSL']))
print('Pearsons correlation:MultipleLines_No,  InternetService_Fiber optic',np.corrcoef(telcom['MultipleLines_No'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:MultipleLines_No,  InternetService_No',np.corrcoef(telcom['MultipleLines_No'],telcom['InternetService_No']))
print('Pearsons correlation:MultipleLines_No, Contract_Month-to-month ',np.corrcoef(telcom['MultipleLines_No'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:MultipleLines_No,  Contract_One year',np.corrcoef(telcom['MultipleLines_No'],telcom['Contract_One year']))
print('Pearsons correlation:MultipleLines_No, Contract_Two year ',np.corrcoef(telcom['MultipleLines_No'],telcom['Contract_Two year']))
print('Pearsons correlation:MultipleLines_No,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['MultipleLines_No'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:MultipleLines_No,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['MultipleLines_No'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:MultipleLines_No,  PaymentMethod_Electronic check',np.corrcoef(telcom['MultipleLines_No'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:MultipleLines_No,  PaymentMethod_Electronic check',np.corrcoef(telcom['MultipleLines_No'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:MultipleLines_No, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['MultipleLines_No'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:MultipleLines_No, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['MultipleLines_No'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:MultipleLines_No, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['MultipleLines_No'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:MultipleLines_No, tenure ',np.corrcoef(telcom['MultipleLines_No'],telcom['tenure']))
print('Pearsons correlation:MultipleLines_No, MonthlyCharges ',np.corrcoef(telcom['MultipleLines_No'],telcom['MonthlyCharges']))
print('Pearsons correlation:MultipleLines_No, TotalCharges ',np.corrcoef(telcom['MultipleLines_No'],telcom['TotalCharges']))


print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:MultipleLines_No phone service, gender ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['gender']))
print('Pearsons correlation:MultipleLines_No phone service, Partner  ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['Partner']))
print('Pearsons correlation:MultipleLines_No phone service, PhoneService ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['PhoneService']))
print('Pearsons correlation:MultipleLines_No phone service, OnlineSecurity ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['OnlineSecurity']))
print('Pearsons correlation:MultipleLines_No phone service, OnlineBackup ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['OnlineBackup']))
print('Pearsons correlation:MultipleLines_No phone service,  DeviceProtection',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['DeviceProtection']))
print('Pearsons correlation:MultipleLines_No phone service,  TechSupport',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['TechSupport']))
print('Pearsons correlation:MultipleLines_No phone service, StreamingTV ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['StreamingTV']))
print('Pearsons correlation:MultipleLines_No phone service, StreamingMovies ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['StreamingMovies']))
print('Pearsons correlation:MultipleLines_No phone service,  PaperlessBilling',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['PaperlessBilling']))
print('Pearsons correlation:MultipleLines_No phone service,  MultipleLines_No',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['MultipleLines_No']))
print('Pearsons correlation:MultipleLines_No phone service,  MultipleLines_No phone service',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:MultipleLines_No phone service,  MultipleLines_Yes',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:MultipleLines_No phone service,  InternetService_DSL',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['InternetService_DSL']))
print('Pearsons correlation:MultipleLines_No phone service,  InternetService_Fiber optic',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:MultipleLines_No phone service,  InternetService_No',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['InternetService_No']))
print('Pearsons correlation:MultipleLines_No phone service, Contract_Month-to-month ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:MultipleLines_No phone service,  Contract_One year',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['Contract_One year']))
print('Pearsons correlation:MultipleLines_No phone service, Contract_Two year ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['Contract_Two year']))
print('Pearsons correlation:MultipleLines_No phone service,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:MultipleLines_No phone service,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:MultipleLines_No phone service,  PaymentMethod_Electronic check',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:MultipleLines_No phone service,  PaymentMethod_Electronic check',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:MultipleLines_No phone service, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:MultipleLines_No phone service, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:MultipleLines_No phone service, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:MultipleLines_No phone service, tenure ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['tenure']))
print('Pearsons correlation:MultipleLines_No phone service, MonthlyCharges ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['MonthlyCharges']))
print('Pearsons correlation:MultipleLines_No phone service, TotalCharges ',np.corrcoef(telcom['MultipleLines_No phone service'],telcom['TotalCharges']))



print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:MultipleLines_Yes, gender ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['gender']))
print('Pearsons correlation:MultipleLines_Yes, Partner  ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['Partner']))
print('Pearsons correlation:MultipleLines_Yes, PhoneService ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['PhoneService']))
print('Pearsons correlation:MultipleLines_Yes, OnlineSecurity ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['OnlineSecurity']))
print('Pearsons correlation:MultipleLines_Yes, OnlineBackup ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['OnlineBackup']))
print('Pearsons correlation:MultipleLines_Yes,  DeviceProtection',np.corrcoef(telcom['MultipleLines_Yes'],telcom['DeviceProtection']))
print('Pearsons correlation:MultipleLines_Yes,  TechSupport',np.corrcoef(telcom['MultipleLines_Yes'],telcom['TechSupport']))
print('Pearsons correlation:MultipleLines_Yes, StreamingTV ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['StreamingTV']))
print('Pearsons correlation:MultipleLines_Yes, StreamingMovies ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['StreamingMovies']))
print('Pearsons correlation:MultipleLines_Yes,  PaperlessBilling',np.corrcoef(telcom['MultipleLines_Yes'],telcom['PaperlessBilling']))
print('Pearsons correlation:MultipleLines_Yes,  MultipleLines_No',np.corrcoef(telcom['MultipleLines_Yes'],telcom['MultipleLines_No']))
print('Pearsons correlation:MultipleLines_Yes,  MultipleLines_No phone service',np.corrcoef(telcom['MultipleLines_Yes'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:MultipleLines_Yes,  MultipleLines_Yes',np.corrcoef(telcom['MultipleLines_Yes'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:MultipleLines_Yes,  InternetService_DSL',np.corrcoef(telcom['MultipleLines_Yes'],telcom['InternetService_DSL']))
print('Pearsons correlation:MultipleLines_Yes,  InternetService_Fiber optic',np.corrcoef(telcom['MultipleLines_Yes'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:MultipleLines_Yes,  InternetService_No',np.corrcoef(telcom['MultipleLines_Yes'],telcom['InternetService_No']))
print('Pearsons correlation:MultipleLines_Yes, Contract_Month-to-month ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:MultipleLines_Yes,  Contract_One year',np.corrcoef(telcom['MultipleLines_Yes'],telcom['Contract_One year']))
print('Pearsons correlation:MultipleLines_Yes, Contract_Two year ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['Contract_Two year']))
print('Pearsons correlation:MultipleLines_Yes,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['MultipleLines_Yes'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:MultipleLines_Yes,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['MultipleLines_Yes'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:MultipleLines_Yes,  PaymentMethod_Electronic check',np.corrcoef(telcom['MultipleLines_Yes'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:MultipleLines_Yes,  PaymentMethod_Electronic check',np.corrcoef(telcom['MultipleLines_Yes'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:MultipleLines_Yes, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:MultipleLines_Yes, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:MultipleLines_Yes, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:MultipleLines_Yes, tenure ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['tenure']))
print('Pearsons correlation:MultipleLines_Yes, MonthlyCharges ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['MonthlyCharges']))
print('Pearsons correlation:MultipleLines_Yes, TotalCharges ',np.corrcoef(telcom['MultipleLines_Yes'],telcom['TotalCharges']))


print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:InternetService_DSL, gender ',np.corrcoef(telcom['InternetService_DSL'],telcom['gender']))
print('Pearsons correlation:InternetService_DSL, Partner  ',np.corrcoef(telcom['InternetService_DSL'],telcom['Partner']))
print('Pearsons correlation:InternetService_DSL, PhoneService ',np.corrcoef(telcom['InternetService_DSL'],telcom['PhoneService']))
print('Pearsons correlation:InternetService_DSL, OnlineSecurity ',np.corrcoef(telcom['InternetService_DSL'],telcom['OnlineSecurity']))
print('Pearsons correlation:InternetService_DSL, OnlineBackup ',np.corrcoef(telcom['InternetService_DSL'],telcom['OnlineBackup']))
print('Pearsons correlation:InternetService_DSL,  DeviceProtection',np.corrcoef(telcom['InternetService_DSL'],telcom['DeviceProtection']))
print('Pearsons correlation:InternetService_DSL,  TechSupport',np.corrcoef(telcom['InternetService_DSL'],telcom['TechSupport']))
print('Pearsons correlation:InternetService_DSL, StreamingTV ',np.corrcoef(telcom['InternetService_DSL'],telcom['StreamingTV']))
print('Pearsons correlation:InternetService_DSL, StreamingMovies ',np.corrcoef(telcom['InternetService_DSL'],telcom['StreamingMovies']))
print('Pearsons correlation:InternetService_DSL,  PaperlessBilling',np.corrcoef(telcom['InternetService_DSL'],telcom['PaperlessBilling']))
print('Pearsons correlation:InternetService_DSL,  MultipleLines_No',np.corrcoef(telcom['InternetService_DSL'],telcom['MultipleLines_No']))
print('Pearsons correlation:InternetService_DSL,  MultipleLines_No phone service',np.corrcoef(telcom['InternetService_DSL'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:InternetService_DSL,  MultipleLines_Yes',np.corrcoef(telcom['InternetService_DSL'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:InternetService_DSL,  InternetService_DSL',np.corrcoef(telcom['InternetService_DSL'],telcom['InternetService_DSL']))
print('Pearsons correlation:InternetService_DSL,  InternetService_Fiber optic',np.corrcoef(telcom['InternetService_DSL'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:InternetService_DSL,  InternetService_No',np.corrcoef(telcom['InternetService_DSL'],telcom['InternetService_No']))
print('Pearsons correlation:InternetService_DSL, Contract_Month-to-month ',np.corrcoef(telcom['InternetService_DSL'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:InternetService_DSL,  Contract_One year',np.corrcoef(telcom['InternetService_DSL'],telcom['Contract_One year']))
print('Pearsons correlation:InternetService_DSL, Contract_Two year ',np.corrcoef(telcom['InternetService_DSL'],telcom['Contract_Two year']))
print('Pearsons correlation:InternetService_DSL,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['InternetService_DSL'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:InternetService_DSL,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['InternetService_DSL'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:InternetService_DSL,  PaymentMethod_Electronic check',np.corrcoef(telcom['InternetService_DSL'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:InternetService_DSL,  PaymentMethod_Electronic check',np.corrcoef(telcom['InternetService_DSL'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:InternetService_DSL, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['InternetService_DSL'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:InternetService_DSL, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['InternetService_DSL'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:InternetService_DSL, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['InternetService_DSL'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:InternetService_DSL, tenure ',np.corrcoef(telcom['InternetService_DSL'],telcom['tenure']))
print('Pearsons correlation:InternetService_DSL, MonthlyCharges ',np.corrcoef(telcom['InternetService_DSL'],telcom['MonthlyCharges']))
print('Pearsons correlation:InternetService_DSL, TotalCharges ',np.corrcoef(telcom['InternetService_DSL'],telcom['TotalCharges']))



print('-------------------------------------------------------------------------------------------------------------')


print ('Pearsons correlation:InternetService_Fiber optic, gender ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['gender']))
print('Pearsons correlation:InternetService_Fiber optic, Partner  ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['Partner']))
print('Pearsons correlation:InternetService_Fiber optic, PhoneService ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['PhoneService']))
print('Pearsons correlation:InternetService_Fiber optic, OnlineSecurity ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['OnlineSecurity']))
print('Pearsons correlation:InternetService_Fiber optic, OnlineBackup ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['OnlineBackup']))
print('Pearsons correlation:InternetService_Fiber optic,  DeviceProtection',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['DeviceProtection']))
print('Pearsons correlation:InternetService_Fiber optic,  TechSupport',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['TechSupport']))
print('Pearsons correlation:InternetService_Fiber optic, StreamingTV ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['StreamingTV']))
print('Pearsons correlation:InternetService_Fiber optic, StreamingMovies ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['StreamingMovies']))
print('Pearsons correlation:InternetService_Fiber optic,  PaperlessBilling',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['PaperlessBilling']))
print('Pearsons correlation:InternetService_Fiber optic,  MultipleLines_No',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['MultipleLines_No']))
print('Pearsons correlation:InternetService_Fiber optic,  MultipleLines_No phone service',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['MultipleLines_No phone service']))
print('Pearsons correlation:InternetService_Fiber optic,  MultipleLines_Yes',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['MultipleLines_Yes']))
print('Pearsons correlation:InternetService_Fiber optic,  InternetService_DSL',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['InternetService_DSL']))
print('Pearsons correlation:InternetService_Fiber optic,  InternetService_Fiber optic',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['InternetService_Fiber optic']))
print('Pearsons correlation:InternetService_Fiber optic,  InternetService_No',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['InternetService_No']))
print('Pearsons correlation:InternetService_Fiber optic, Contract_Month-to-month ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['Contract_Month-to-month']))
print('Pearsons correlation:InternetService_Fiber optic,  Contract_One year',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['Contract_One year']))
print('Pearsons correlation:InternetService_Fiber optic, Contract_Two year ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['Contract_Two year']))
print('Pearsons correlation:InternetService_Fiber optic,  PaymentMethod_Bank transfer (automatic)',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['PaymentMethod_Bank transfer (automatic)']))
print('Pearsons correlation:InternetService_Fiber optic,  PaymentMethod_Credit card (automatic)',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['PaymentMethod_Credit card (automatic)']))
print('Pearsons correlation:InternetService_Fiber optic,  PaymentMethod_Electronic check',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:InternetService_Fiber optic,  PaymentMethod_Electronic check',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['PaymentMethod_Electronic check']))
print('Pearsons correlation:InternetService_Fiber optic, tenure_group_Tenure_0-24 ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['tenure_group_Tenure_0-24']))
print('Pearsons correlation:InternetService_Fiber optic, tenure_group_Tenure_24-48 ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['tenure_group_Tenure_24-48']))
print('Pearsons correlation:InternetService_Fiber optic, tenure_group_Tenure_48-72 ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['tenure_group_Tenure_48-72']))
print('Pearsons correlation:InternetService_Fiber optic, tenure ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['tenure']))
print('Pearsons correlation:InternetService_Fiber optic, MonthlyCharges ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['MonthlyCharges']))
print('Pearsons correlation:InternetService_Fiber optic, TotalCharges ',np.corrcoef(telcom['InternetService_Fiber optic'],telcom['TotalCharges']))




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
X_copy = X.copy()

#
sns.distplot(X.gender,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.StreamingTV,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.SeniorCitizen,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.Partner,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.Dependents,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.PhoneService,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.OnlineSecurity,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.OnlineBackup,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.DeviceProtection ,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.TechSupport ,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.StreamingTV ,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.StreamingMovies ,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.PaperlessBilling  ,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.MultipleLines_No ,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.InternetService_DSL,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.MultipleLines_Yes ,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.InternetService_No ,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.tenure,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.MonthlyCharges ,hist=False,rug=True,bins=telcom['Churn'])
plt.show()
sns.distplot(X.TotalCharges ,hist=False,rug=True,bins=telcom['Churn'])
plt.show()



import statsmodels.api as stf
lm = stf.OLS(y_train,X_train)
model = lm.fit()
print('satisfs')
print(model.summary())


# del X['MultipleLines_No phone service']
# del X['MultipleLines_No']
# del X['MultipleLines_Yes']
# del X['gender']
# del X['PaymentMethod_Credit card (automatic)']
# del X['PaymentMethod_Bank transfer (automatic)']
# del X['MonthlyCharges'] # çarpıklık
# del X['tenure']
# del X['TotalCharges']
# del X['PhoneService']
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

bestfeatures = SelectKBest(score_func=chi2,  k = 'all')
fit = bestfeatures.fit(X,y)

dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['parameters','Churn']  #naming the dataframe columns
print("Feature Selection")
print(featureScores.nlargest(33,'Churn'))  #print 10 best features

a = pd.DataFrame()
b = pd.DataFrame()
# a = X_copy[['gender','SeniorCitizen','Dependents','PhoneService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport','StreamingTV','StreamingMovies','PaperlessBilling',
#             'MultipleLines_No','MultipleLines_No phone service','MultipleLines_Yes']]
# sns.heatmap(a.corr(), annot = True, fmt='.1g')
# plt.show()
# b = X_copy[['InternetService_DSL','InternetService_Fiber optic','InternetService_No','Contract_Month-to-month','Contract_One year','Contract_Two year',
#             'PaymentMethod_Electronic check','tenure_group_Tenure_0-24','tenure_group_Tenure_24-48','tenure_group_Tenure_48-72',]]
# sns.heatmap(b.corr(),annot=True,fmt='.2g')
# plt.show()



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


