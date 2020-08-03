import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# import pandas_profiling
# from pandas_profiling import ProfileReport
import warnings
from sklearn import preprocessing

input_path = r"C:\Users\hlltk\PycharmProjects\churn_analysis"
input_filename = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


df = pd.read_csv(input_path+"/"+input_filename,sep=',')
df.head()
print(df)
print(df.dtypes)
print(df.isnull().sum())#boş olan değerler

summary = df.describe(include=["O"])
print(summary)
df.dropna(how="any",inplace=True)
sum = df.isnull().sum()
print(sum)

for item in df.columns:
    print(item)
    print (df[item].unique())
y = df["Churn"].value_counts()
sns.barplot(y.index, y.values)
print(sns.barplot(y.index, y.values))
# import matplotlib.pyplot as plt
# plt.show()
label_encoder = preprocessing.LabelEncoder()
one_hot_encoder = preprocessing.OneHotEncoder(categorical_features = [0])
lb =preprocessing.LabelBinarizer()

customerID = df.iloc[:,0].values
customerID = pd.DataFrame(customerID)
SeniorCitizen = df.iloc[:,2].values
SeniorCitizen = pd.DataFrame(SeniorCitizen)
gender = label_encoder.fit_transform(df['gender'])
Partner = label_encoder.fit_transform(df['Partner'])
Dependents = label_encoder.fit_transform(df['Dependents'])
tenure = df.iloc[:,5].values
tenure = pd.DataFrame(tenure)
print(tenure.shape)
PhoneService = label_encoder.fit_transform(df['PhoneService'])
print(PhoneService.shape)

#df['MultipleLines'] = label_encoder.fit_transform(df['MultipleLines'])

multiplelines = pd.get_dummies(df.iloc[:,7].values)
multiplelines = pd.DataFrame(multiplelines)
InternetService = df.iloc[:,8].values

InternetService = pd.get_dummies(InternetService)
InternetService = pd.DataFrame(InternetService)

OnlineSecurity = df.iloc[:,9].values

OnlineSecurity = pd.get_dummies(OnlineSecurity)
OnlineSecurity = pd.DataFrame(OnlineSecurity)

OnlineBackup = pd.get_dummies(df.iloc[:,10].values)
OnlineBackup = pd.DataFrame(OnlineBackup)

DeviceProtection = pd.get_dummies(df.iloc[:,11].values)
DeviceProtection = pd.DataFrame(DeviceProtection)

TechSupport = pd.get_dummies(df.iloc[:,12].values)
print("TechSupport",TechSupport)
TechSupport = pd.DataFrame(TechSupport,columns = ['No' , 'No internet service' , 'Yes'])

StreamingTV = pd.get_dummies(df.iloc[:,13].values)
StreamingTV = pd.DataFrame(StreamingTV)

StreamingMovies = pd.get_dummies(df.iloc[:,14].values)
StreamingMovies = pd.DataFrame(StreamingMovies)

Contract = pd.get_dummies(df.iloc[:,15].values)
Contract = pd.DataFrame(Contract)

PaperlessBilling = label_encoder.fit_transform(df.iloc[:,16].values)
PaperlessBilling = pd.DataFrame(PaperlessBilling)

PaymentMethod = pd.get_dummies(df.iloc[:,17].values)
PaymentMethod = pd.DataFrame(PaymentMethod)

MonthlyCharges = df.iloc[:,18].values
MonthlyCharges = pd.DataFrame(MonthlyCharges)

TotalCharges = df.iloc[:,19].values
TotalCharges = pd.DataFrame(TotalCharges)

Churn = df.iloc[:,20].values
Churn = pd.DataFrame(Churn)

gender = pd.DataFrame(gender)
Partner = pd.DataFrame(Partner)

a=gender.join(Partner,lsuffix='gender', rsuffix='Partner')
print(a)


customerID,SeniorCitizen, tenure ,
#                           MonthlyCharges, TotalCharges

x = customerID.join(SeniorCitizen,lsuffix='customerID', rsuffix='SeniorCitizen')
print(x)
y = tenure.join(MonthlyCharges,lsuffix='tenure', rsuffix='MonthlyCharges')

xy =x.join(y)


Dependents = pd.DataFrame(Dependents)
PhoneService = pd.DataFrame(PhoneService)
b=Dependents.join(PhoneService,lsuffix='Dependents', rsuffix='PhoneService')
print(b)

c = multiplelines.join(InternetService,lsuffix='multiples', rsuffix='InternetSevices')
print(c)

ab = a.join(b)
print("------")
print("ab",ab)

abc = ab.join(c)
print("abc",abc)

d = OnlineBackup.join(DeviceProtection,lsuffix='OnlineBackup', rsuffix='DeviceProtection')
print(d)

abcd = abc.join(d)
print("abcd",abcd)


e = TechSupport.join(StreamingTV,lsuffix='TechSupport', rsuffix='StreamingTv')
print("e",e)

abcde = abcd.join(e)
print("abcde",abcde)

f = StreamingMovies.join(Contract,lsuffix='StreamingMovies', rsuffix='Contract')
print("f",f)
abcdef = abcde.join(f,lsuffix='abcde', rsuffix='f')
print("abcdef",abcdef)

g = PaperlessBilling.join(PaymentMethod)
print("g",g)

abcdefg = abcdef.join(g)
print("abcefg",abcdefg)

features = xy.join(abcdefg)
print(features)


