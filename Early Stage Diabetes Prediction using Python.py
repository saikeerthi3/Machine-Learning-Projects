#!/usr/bin/env python
# coding: utf-8

# # Loading the data:

# In[1]:


import numpy as np
import pandas as pd
data1=pd.read_csv('C:/Users/saike/OneDrive/文档/diabetes_data.csv')
data1


# # obtaining the first five rows of data:

# In[2]:


data1.head()


# In[3]:


#checking the dimensions of the data:

data1.shape


# # Data Pre-Processing:

# In[4]:


#checking for null values in the data:

data1.isna().sum()


# In[5]:


new_column_names = {'sudden weight loss':'sudden_weight_loss','delayed healing': 'delayed_healing','class':'diabetes'}

# Using the rename method to change column names
data1.rename(columns=new_column_names, inplace=True)

data1


# In[6]:


#Converting the categorical variables to numerical values:

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
def label_encode_columns(data1, columns):
    le = LabelEncoder()
    for column in columns:
        data1[column] = le.fit_transform(data1[column])
    return data1

# Specifying the columns to be label encoded
columns_to_encode = ['Gender', 'Polyuria','Polydipsia','sudden_weight_loss','weakness',
                     'Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed_healing',
                     'partial paresis','muscle stiffness','Alopecia','Obesity','diabetes']
# Performing label encoding
df_encoded = label_encode_columns(data1,columns_to_encode)
df_encoded


# # Descriptive and Exploratory Data Analysis:

# In[7]:


#obtaining descriptive information of the data:

data1.describe()


# In[8]:


#obtaining number of times a  specific class is present:

count_diabetes = data1['diabetes'].value_counts()
count_diabetes


# In[9]:


#obtaining plot of diabetes with each class:

count_diabetes.plot(kind="bar", color=["red", "lightblue"])


# In[10]:


#obtaining histogram of 'Age':

import matplotlib.pyplot as plt
data1.hist(['Age'],figsize=(12, 10), bins=20)
plt.xlabel('Age in years')
plt.ylabel('Frequency')
plt.show()


# In[11]:


#obtaining the Box plot of 'Age':

import matplotlib.pyplot as plt
plt.boxplot(data1['Age'])
plt.xlabel('Age')
plt.ylabel('Years')
plt.title('Boxplot of Age of Patients')
plt.show()


# In[12]:


#obtaining the range of variables:

categorical_value = []
continous_value = []
for column in data1.columns:
    print('==============================')
    print(f"{column} : {data1[column].unique()}")
    if len(data1[column].unique()) <= 10:
        categorical_value.append(column)
    else:
        continous_value.append(column)


# In[13]:


#obtaining a plot showing patients of which age have highr chances of diabetes:
plt.figure(figsize=(15, 15))

for i, column in enumerate(continous_value, 1):
    plt.subplot(3, 2, i)
    data1[data1["diabetes"] == 0][column].hist(bins=35, color='blue', label='Have Diabetes = NO')
    data1[data1["diabetes"] == 1][column].hist(bins=35, color='red', label='Have Diabetes = YES')
    plt.legend()
    plt.xlabel(column)
    


# In[14]:


#obtaining the histograms of all the categorical variables:

data1.hist(figsize=(20,20))
plt.show()


# In[15]:


#adding random values to the data to obtain a scatter plot:
import seaborn as sns

jittered_data = data1 + np.random.normal(0, 0.1, data1.shape)

# Creating scatter plot
sns.scatterplot(x='Polyuria', y='diabetes', data=jittered_data, marker='o', alpha=0.7)

plt.xlabel('Polyuria')
plt.ylabel('Diabetes')
plt.title('Scatter Plot between Polyuria and Diabetes')


# In[16]:


sns.scatterplot(x='Age', y='diabetes', data=data1, marker='o', alpha=0.7)

plt.xlabel('Age')
plt.ylabel('Diabetes')
plt.title('Scatter Plot between Age and Diabetes')


# In[17]:


sns.scatterplot(x='Polydipsia', y='diabetes', data=jittered_data, marker='o', alpha=0.7)

plt.xlabel('Polydipsia')
plt.ylabel('Diabetes')
plt.title('Scatter Plot between Polydipsia and Diabetes')


# In[18]:


sns.scatterplot(x='Alopecia', y='diabetes', data=jittered_data, marker='o', alpha=0.7)

plt.xlabel('Alopecia')
plt.ylabel('Diabetes')
plt.title('Scatter Plot between Alopecia and Diabetes')


# # Scaling the variable 'Age':

# In[19]:


from sklearn.preprocessing import StandardScaler

s_sc = StandardScaler()
col_for_scaling = ['Age']
data1[col_for_scaling] = s_sc.fit_transform(data1[col_for_scaling])
data1


# # Model1 - Classification using Logistic Regression:

# In[20]:


#importing necessary metrics from sklearn:

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#defining a function to obtain calssification report:

def print_score(clf, X_train, Y_train, X_test, Y_test, train=True):
    if train:
        train_pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(Y_train,train_pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(Y_train, train_pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(Y_train, train_pred)}\n")
        sns.heatmap(confusion_matrix(Y_train, train_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix of Train Result')
        plt.show()
         
    elif train==False:
        test_pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(Y_test, test_pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(Y_test,test_pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(Y_test,test_pred)}\n")
        sns.heatmap(confusion_matrix(Y_test, test_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix of Test Result')
        plt.show()


# In[21]:


from sklearn.model_selection import train_test_split

###dropping the target variable from the dataset:
X = data1.drop('diabetes', axis=1)
###creating a variable Y with only target variable :
Y = data1.diabetes
### splitting the data into train and test parts with 70% of data for training and 30% of data for testing:
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# In[22]:


from sklearn.linear_model import LogisticRegression

#performing logistic Regression model on data:

lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(X_train, Y_train)

print_score(lr_clf, X_train, Y_train, X_test, Y_test, train=True)
print_score(lr_clf, X_train, Y_train, X_test, Y_test, train=False)


# In[23]:


#obtaining accuracy scores of training and testing of model:

test_score = accuracy_score(Y_test, lr_clf.predict(X_test)) * 100
train_score = accuracy_score(Y_train, lr_clf.predict(X_train)) * 100

results = pd.DataFrame(data=[["Logistic Regression", train_score, test_score]], 
                          columns=['Model', 'Training Accuracy %', 'Testing Accuracy %'])
results


# # Model2 - Classification using Decision Tree:

# In[24]:


from sklearn import tree
decisiontree1 = tree.DecisionTreeClassifier(random_state = 42)
decisiontree_model = decisiontree1.fit(X_train, Y_train)
X_train_res=decisiontree_model.predict(X_train)

# Predicting the target for the test set
target_pred = decisiontree_model.predict(X_test)


# In[25]:


###calssification report of train result
print("Train Result:\n================================================")
clf_report1 = pd.DataFrame(classification_report(Y_train,X_train_res, output_dict=True))
clf_report1


# In[26]:


#obtaining training accuracy:

train_accuracy = accuracy_score(Y_train,X_train_res)*100
train_accuracy


# In[27]:


#obtaining confusion matrix of train result:

confusion_matrix(Y_train,X_train_res)
sns.heatmap(confusion_matrix(Y_train,X_train_res ), annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])


# In[28]:


##classification report of test results:
print("Test Result:\n================================================")
clf_report = pd.DataFrame(classification_report(Y_test, target_pred, output_dict=True))
clf_report


# In[29]:


# Calculate testing accuracy
test_accuracy = accuracy_score(Y_test, target_pred)*100
test_accuracy


# In[30]:


#obtaining confusion matrix of test result:

confusion_matrix(Y_test,target_pred)
sns.heatmap(confusion_matrix(Y_test, target_pred), annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])


# In[ ]:





# In[ ]:





# In[ ]:




