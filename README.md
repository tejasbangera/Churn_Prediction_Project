# Udacity Capstone Churn Prediction project

This is Udacity Azure Machine Learning Nanodegree Capstone Project

In this project, I'm working with the Telco Customer Churn dataset and the goal is to predict if the customer will churn or not. i.e., a Classification problem. 
We will clean the dataset using the clean_data function in train.py (data preparation script) and then the data set is split into train and test sets. 
Our classification algorithm is Logistic Regression and the target variable that we want to predict is categorical and has only two possible outcomes: churn or not churn and  then we will deploy the best model and consume it.

## Dataset

**Overview**

What is Churn? 

Churn is a process in which customers stop or plan to stop using services/contracts of a company. So churn prediction is about identifying customers who are likely to cancel their services/contracts soon. Then companies can offer discounts or other benefits on these services and users can continue with their services.
Naturally, we can use the past data about customers who churned and based on that we will create a model for identifying present customers who are about to go away. 
Some of our customers are churning. They no longer are using our services and going to a different provider. We would like to prevent that from happening. For that, we develop a system for identifying these customers and company can offer them an incentive to stay.

Dataset columns:

CustomerID â€” the ID of the customer

Gender â€” male/female

Senior Citizen â€” whether the customer is a senior citizen (0/1)

Partner â€” whether they live with a partner (yes/no)

Dependents â€” whether they have dependents (yes/no)

Tenure â€” number of months since the start of the contract

Phone service â€” whether they have phone service (yes/no)

Multiple lines â€” whether they have multiple phone lines (yes/no/no phone service)

Internet service â€” the type of internet service (no/fiber/optic)

Online security â€” if online security is enabled (yes/no/no internet)

Online backup â€” if online backup service is enabled (yes/no/no internet)

Device protection â€” if the device protection service is enabled (yes/no/no internet)

Tech support â€” if the customer has tech support (yes/no/no internet)

Streaming TV â€” if the TV streaming service is enabled (yes/no/no internet)

Streaming movies â€” if the movie streaming service is enabled (yes/no/no internet)

Contract â€” the type of contract (monthly/yearly/two years)

Paperless billing â€” if the billing is paperless (yes/no)

Payment method â€” payment method (electronic check, mailed check, bank transfer, credit card)

Monthly charges â€” the amount charged monthly (numeric)

Total charges â€” the total amount charged (numeric)

**Churn â€” if the client has canceled the contract (yes/no) and this is the target column for our prediction**


Project main steps:
In this project, we will be following the below steps:

**Authentication:**
In this step, we need to create a Security Principal (SP) to interact with the Azure Workspace.Here IÃ¢â‚¬â„¢m using the lab Udacity provided to me, so I have skipped this step as I'm are not authorized to create a security principal.

First, we need to download the Churn dataset from Kaggle. Link: https://www.kaggle.com/blastchar/telco-customer-churn

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/Churn%20dataset.png)

**HyperParameter Tuning**

In this step, we create an experiment, configure a compute cluster, and use that cluster to run the experiment using HyperParameter Tuning.
Hyperparameters are -C (Inverse of regularization strength. Smaller values cause stronger regularization) and --max_iter (Maximum number of iterations to converge)

Azure Machine Learning supports three types of sampling: Random sampling, Grid sampling and Bayesian sampling.

We have used Random sampling as Random sampling supports discrete and continuous hyperparameters. It supports early termination of low-performance runs and values are chosen randomly as a result of which it is not much compute intensive and works perfect for project

Grid sampling supports discrete hyperparameters. Use grid sampling if you can budget to exhaustively search over the search space. Supports early termination of low-performance runs but it is compute intensive as compared to Random sampling

Bayesian sampling is based on the Bayesian optimization algorithm. It picks samples based on how previous samples did, so that new samples improve the primary metric. Bayesian sampling is recommended if you have enough budget to explore the hyperparameter space. For best results, we recommend a maximum number of runs greater than or equal to 20 times the number of hyperparameters being tuned. The number of concurrent runs has an impact on the effectiveness of the tuning process. A smaller number of concurrent runs may lead to better sampling convergence, since the smaller degree of parallelism increases the number of runs that benefit from previously completed runs. Bayesian sampling only supports choice, uniform, and uniform distributions over the search space. So, this method is bit compute intensive for our project.

Azure Machine Learning supports four early termination policies - Bandit policy, Median stopping policy, Truncation selection policy and No termination policy. We have used Bandit policy as bandit terminates runs where the primary metric is not within the specified slack factor/slack amount compared to the best performing run. This allows to compute to spend more time on the other hyper parameters that matters.

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS01.png)

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS02.png)

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS03.png)

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS06.png)

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS04.png)

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS05.png)

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS07.png)

**Automated ML Experiment**

This images illustrates how Auto ML works:

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/automl-concept-diagram2.png)

In this step, we create an experiment using Automated ML and using the same compute cluster created earlier to run the experiment.

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS10.png)
![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/S11.5.png)

The best algorithm found is MaxAbsScaler, XGBoostClassifier with an accuracy of 80.6%.

We will deploying the best AutoML model as its accuracy is better than hyper-parameter model.

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS11..png)
![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS13.png)

Swagger is a tool that eases the documentation efforts of HTTP APIs. It makes it easier to explain what type of GET and POST requests. 
To interact with the model obtained we deployed it using Azure Container Instance (ACI) then we can consume our model using Swagger and we also enabled application insights to view the logs generated.

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS11.png)
![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/SS12.png)

Finally, we can now interact with the model and feed some test data to the script and running it. So once a model has been deployed, an endpoint will be available which allows user to send inputs to the trained model and get a response back. This is called as consuming deployed service.

![image](https://github.com/tejasbangera/Udacity-Capstone-Project/blob/main/Images/S15.png)

Screen Recording:

Screen Recording Link: https://youtu.be/Dm_TDT5zpho

Future Improvements:

We can try with Deep Learning for better performance.

Resampling or adding more data can resolve class-imbalance issue and improve accuracy.
