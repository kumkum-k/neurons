#for numeric computing
import numpy as np
#for dataframe
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Ignore Warnings
import warnings
warnings.filterwarnings("ignore")


#Machine Learning Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.svm import SVC,LinearSVC
#Regression
from sklearn.linear_model import LinearRegression,Ridge,Lasso,RidgeCV,ElasticNet,LogisticRegression
from sklearn.ensemble import RandomForestRegressor,BaggingRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.svm import SVR

#Modelling Helpers
from sklearn.preprocessing import Normalizer,scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV,KFold,cross_val_score,ShuffleSplit,cross_validate

#Preprocessing
from sklearn.preprocessing import MinMaxScaler,StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer

#Classification
from sklearn.metrics import accuracy_score,recall_score,f1_score,fbeta_score,r2_score,roc_auc_score,roc_curve,auc,cohen_kappa_score

#to display all the interactive output without using print()
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactive="all"
df=pd.read_csv('Toddler Autism dataset July 2018.csv')
df.info()
df.head(20)
df.shape

df.describe()

print("COLUMNS")
df.columns
#remove unwanted columns
df.drop(['Case_No','Who completed the test'],axis=1,inplace=True)
df.columns

#Calculating the percentage of babies shows the symptoms of autism
yes_autism = df[df['Class/ASD Traits '] == 'Yes']
no_autism = df[df['Class/ASD Traits '] == 'No']
print("Toddlers:",round(len(yes_autism)/len(df)==100,2))
print("Toddlers:",round(len(no_autism)/len(df)==100,2))

#Displaying The Content of the target column
df['Class/ASD Traits '].value_counts()

fig=plt.gcf()
fig.set_size_inches(7,7)
plt.pie(df['Class/ASD Traits '].value_counts(),labels=('no_autism','yes_autism'),explode=[0.1,0],autopct='%1.1f%%',shadow=True,startangle=90,labeldistance=1.1)
plt.show()


#Checking null data
#df.isnull().sum()
#df.dtypes
#corr=df.corr()
#plt.figure(figsize=(15,15))

#displaying the no. of positive cases of ASD with regared Ethnicity
yes_autism['Ethnicity'].value_counts()

#removing 'Qchat-10-score'
df.drop('Qchat-10-Score',axis=1,inplace=True)

le=LabelEncoder()
columns=['Ethnicity','Family_mem_with_ASD','Class/ASD Traits ','Sex','Jaundice']

for col in columns:
    df[col]=le.fit_transform(df[col])
df.dtypes

df.head(25)
X=df.drop(['Class/ASD Traits '],axis=1)
y=df['Class/ASD Traits ']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.40,random_state=42)
X.isnull().sum()
X.info()

models=[]
models.append(('LR',LogisticRegression()))
models.append(('SVC',SVC()))
models.append(('RF',RandomForestRegressor()))
models.append(('XGB',XGBClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('CART',DecisionTreeClassifier()))
for name,model in models:
    model.fit(X_train,y_train)
    y_hat_test=model.predict(X_test).astype(int)
    y_hat_train=model.predict(X_train).astype(int)
    print(name,'Accuracy Score is : ',round(accuracy_score(y_test,y_hat_test)))


#regression
modelks=[]
modelks.append(('LR',LogisticRegression()))
modelks.append(('SVC',SVC()))
modelks.append(('RF',RandomForestRegressor()))
modelks.append(('XGB',XGBClassifier()))
modelks.append(('KNN',KNeighborsClassifier()))
modelks.append(('CART',DecisionTreeClassifier()))
for name,model in modelks:
    model.fit(X_train,y_train)
    y_hat_test=model.predict(X_test).astype(int)
    y_hat_train=model.predict(X_train).astype(int)
    print(name,'Accuracy Score is : ',round(accuracy_score(y_test,y_hat_test)))
    
for name,modepl in modelks:
    
    y_hat_test=modepl.predict(X_test).astype(int)
    y_hat_train=modepl.predict(X_train).astype(int)
    print(name,'Accuracy Score is : ',round(accuracy_score(y_test,y_hat_test),2))

for name,model in models:
    y_hat_test=model.predict(X_test).astype(int)
    y_hat_train=model.predict(X_train).astype(int)
    print(name,'Accuracy Score is : ',round(accuracy_score(y_test,y_hat_test),2))

svc=SVC()
params={
    'C':[0.1,0.8,0.9,1,1.1,1.2,1.3,1.4],
    'kernel':['linear','rbf'],
    'gamma':[0.1,0.8,0.9,1,1.2,1.2,1.3,1.4]
}
clf=GridSearchCV(svc,param_grid=params,scoring='accuracy',cv=10,verbose=2)
clf.fit(X_train,y_train)
clf.best_params_

#re-running the MOdel with best parameters
svc1=SVC(C=0.8,gamma=0.1,kernel='linear')
svc1.fit(X_train,y_train)
y_hat_test=svc1.predict(X_test)
print(accuracy_score(y_test,y_hat_test))

svcgrid_test_acc=round(accuracy_score(y_test,y_hat_test),2)
svcgrid_test_acc

# Random Forest (grid search and pipeline)
# Instantiate  the pippeline
from sklearn.pipeline import Pipeline
pipe=Pipeline([('classifier',RandomForestClassifier(random_state=123))])

grid=[{'classifier__criterion':['gini','entropy'],
'classifier__n_estimators':[10,20,50,100],
'classifier__max_depth':[None,5,3,10],
'classifier__min_samples_split':[1.0,6,10],
'classifier__min_samples_leaf':[1,6,10],
'classifier__class_weight':['balanced']}]

# from sklearn.model_selection import GridSearchCV

clf=GridSearchCV(estimator=pipe,param_grid=grid,cv=5,scoring='roc_auc',n_jobs=-1)
clf.fit(X_train,y_train)
y_hat_train=clf.predict(X_train)
y_hat_test=clf.predict(X_test)
#research best estimator from grid
best_clf_estimator=clf.best_estimator_
best_clf_estimator.fit(X_train,y_train)
#Prediction for FR
y_hat_train=best_clf_estimator.predict(X_train)
y_hat_test=best_clf_estimator.predict(X_test)
rf_gridsearch_test_acc=round(accuracy_score(y_test,y_hat_test),2)
print("The Accuracy Of Random Forest ALgorithm After Grid Search View Is : ",rf_gridsearch_test_acc)

print(round(clf.score(X_train,y_train)))
print(round(clf.score(X_test,y_test)))

clf.best_params_

from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam  # Import Adam optimizer

model = Sequential()
# #Define Input Layer with 15 features as input
model.add(Dense(100, input_dim=15, activation='relu'))
# #Single Output Layer With One Neuron Since v only waant to predict 2 classes yes autism = 1 or no autism =m 0 
model.add(Dense(activation='sigmoid', units=1))

 #Compile the Neural Network
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.0001),  # Specify the learning rate directly
    metrics=['acc'])


# Fit the model to training data
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

model.summary()



#Structureing of keras neuralnetwork model
# from keras.utils.vis_utils import plot_model
# plot_model(model)
from tensorflow.keras.utils import plot_model

# ...

# Now use plot_model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)

#looking at the Structure of neural network

#predictions 
predict_x=model.predict(X_test)
y_hat_test=np.argmax(predict_x,axis=1)
predict_y=model.predict(X_train)
y_hat_train=np.argmax(predict_x,axis=1)
print("Trained")

#Create Classifier Summary Table
LogisticRegression_Accuracy=1.0
LinearDiscriminantAnalysis_accuracy=0.96
KNeighborsClassifier_accuracy=0.91
DecisionTreeClassifier_accuracy=0.91
GaussianNB_accuracy=0.94
SVC_beforegrid_accuracy=0.78
RandomForest_beforegrid_accuracy=0.64
XGBClassifier_accuracy=0.99
GradientBooosting_accuracy=0.64
AdaBoosting_accuracy=0.49
SVC_aftergrid_accuracy=1.0
RandomForest_aftergrid_accuracy=0.96
Neuralnetwork_SKLearn_accuracy=0.99
Neuralnetwork_Keras_accuracy=0.95

models=['LogisticRegression',
'KNeighborsClassifier',
'SVC_beforegrid_accuracy',
'RandomForest_beforegrid',
'SVC_aftergrid',
'RandomForest_aftergrid',
'Neuralnetwork_SKLearn Accuracy',
'Neuralnetwork_Keras'
]
test_Accuracy=[1.0,0.91,0.78,0.64,1.0,0.96,0.99,0.95]
accuracy_summary=pd.DataFrame([models,test_Accuracy]).T
accuracy_summary.columns=['Classifier','test_Accuracy']

accuracy_summary

import pickle

#Save Trained Model To File
pickle.dump(best_clf_estimator,open("autism.pkl","wb"))

loaded_model=pickle.load(open("autism.pkl","rb"))
loaded_model.predict(X_test)
loaded_model.score(X_test,y_test)

X_test.head(1)

import pickle
import pandas as pd

# Load the trained model
loaded_model = pickle.load(open("autism.pkl", "rb"))

# User inputs (as mentioned in the previous code snippet)
A1 = int(input("Does your child look at you when you call his/her name? (0: No, 1: Yes)"))
A2 = int(input("How easy is it for you to get eye contact with your child? (0: Very easy, 1: Quite easy, 2: Quite difficult, 3: Very difficult, 4: Impossible)"))
A3=int(input())
A4=int(input())
A5=int(input())
A6=int(input())
A7=int(input())
A8=int(input())
A9=int(input())
A10=int(input())


Age_Mons = int(input("Enter your child's age in months: "))
Sex = int(input("Enter your child's gender (0: Female, 1: Male): "))
Ethnicity = int(input("Enter your child's ethnicity: "))
Jaundice = int(input("Was your child born with jaundice? (0: No, 1: Yes)"))
Family_mem_with_ASD = int(input("Do you have family members with ASD? (0: No, 1: Yes)"))

# Prepare the user input in a DataFrame format
user_data = pd.DataFrame({
    'A1': [A1],
    'A2': [A2],
    'A3': [A3],
    'A4': [A4],
    'A5': [A5],
    'A6': [A6],
    'A7': [A7],
    'A8': [A8],
    'A9': [A9],
    'A10': [A10],
    'Age_Mons': [Age_Mons],
    'Sex': [Sex],
    'Ethnicity': [Ethnicity],
    'Jaundice': [Jaundice],
    'Family_mem_with_ASD': [Family_mem_with_ASD]
})

# Get probabilities for each class (0 - No ASD, 1 - ASD)
proba = loaded_model.predict_proba(user_data)[0]

# Display the percentage likelihood of ASD
percentage_asd = proba[1] * 100  # Percentage likelihood of ASD (class 1)
print(f"The percentage likelihood of ASD is: {percentage_asd:.2f}%")