## Problem Statement:

Customer churn is the percentage of customers who stop using a company's products or services during a certain time frame. It is a critical metric because a high churn rate can significantly impact the profitability and long-term success of a business. This project aims to develop a predictive model that accurately determines the likelihood of a bank customer discontinuing services within the next six months. The model will help the bank deploy effective retention strategies, ultimately preserving revenue and enhancing customer satisfaction.

![image](https://github.com/user-attachments/assets/af4f7ebd-0423-42a2-bc31-97f9de1ff8d0)


### Dataset Description
The dataset consists of 10,000 customer records, each featuring demographic information, account details, and other relevant attributes that could influence a customer's decision to leave the bank. There are 14 initial attributes (or columns) for each customer, which include both identifiers and potential predictors of churn.

link to the dataset: https://www.kaggle.com/datasets/barelydedicated/bank-customer-churn-modeling

### Attributes:

RowNumber: A numerical identifier for the row in the dataset.

CustomerId: A unique identifier for each customer.

Surname: The last name of the customer.

CreditScore: A score assigned to a customer based on their credit history.

Geography: The country of the bank's branch where the customer's account is located.

Gender: The customer's gender (male/female).

Age: The customer's age in years.

Tenure: The number of years the customer has been with the bank.

Balance: The account balance maintained by the customer.
NumOfProducts: The number of banking products used by the customer.

HasCrCard: Indicates whether the customer has a credit card with the bank (1 for Yes, 0 for No).

IsActiveMember: Indicates whether the customer is considered an active member based on their account activity (1 for Yes, 0 for No).

EstimatedSalary: The estimated annual salary of the customer.
Exited: Indicates whether the customer has churned (1 for Yes, 0 for No). This is the target variable for our predictive model.

Steps involved:

### 1.Data Preprocessing:

It is one of most important steps to be followed. Pre Processing of the data helps to maintain accuracy and remove inconsistancies in the data. During preprocessing, the identifier columns (RowNumber, CustomerId, Surname) were removed as they are not relevant for the predictive modeling process. The remaining features were used to build the predictive model.

### EDA (Exploratory Data Analysis):

EDA is crucial in the data preprocessing stage of any data analysis or machine learning project. It helps in understanding the distributions of the columns, find any outliers, null values etc. 

### Model Development:

The goal is to build a predictive model that can accurately classify customers who are likely to churn. The model development process includes the following steps:

Data Splitting: Dividing the dataset into training and testing sets to evaluate the model's performance.

Model Training: Using machine learning algorithms to train the model on the training data.

Model Evaluation: Assessing the model's performance using metrics such as accuracy, precision, recall, and F1-score.

Model Deployment: Implementing the model in a real-world setting to make predictions on new data.


