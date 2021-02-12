
*NOTE:* This file is a template that you can use to create the README for your project. The *TODO* comments below will highlight the information you should be sure to include.

# Telco Customer Churn Prediction 

*TODO:* This project is about predicting customer churn for telco customers. It used the dataset from Kaggle(https://www.kaggle.com/blastchar/telco-customer-churn). It is a classification problem which focuses on predicting whether or not the customers left within the last month (this value is called Churn and it has two values: Yes or No. )
## Dataset
### Overview
*TODO*: Explain about the data you are using and where you got it from.

The data set includes information about: 
Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
Demographic info about customers – gender, age range, and if they have partners and dependents
Customers who left within the last month – this is the column is called Churn which has to be predicted.


### Task
*TODO*: Explain the task you are going to be solving with this dataset and the features you will be using for it.
The task that is going to be solved using this dataset is prediction of Customer Churn using the the rest of the features of that customer. In the train.py file, the dataset is first cleaned by dropping unnecessary columns like  "gender","MultipleLines" and "PaymentMethod" etc.

### Access
*TODO*: Explain how you are accessing the data in your workspace.

The data is downloaded from Kaggle and uploaded to a github repository whose link is provided in the notebook, then using TabularDatasetFactory method, it is converted to tabular format, and then it is registered with the key of “data” and uploaded to Azure workspace.


## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment



In AutoML config, we set the experiment timeout in minutes to be 30, iterations to be 100, maximum concurrent iterations to 8 and number of cross-validations to be 5. Task is classification and primary metric is accuracy. Since 5 folds cross validations are being performed, so for each training we use 4/5 th of data and 1/5 th of data is used in each validation with a different holdout fold each time.  The label column is “Churn”.



### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?
The  top contender if AutoML is Voting Ensemble model with accuracy of 80.9%. The model can be improved by increasing the number of iterations, changing the timeout to let the automl run for more than 30 minutes and using neural networks for classification.

![AutoML Results:](images/1.PNG)
![AML Run with Metric](images/2.PNG) 
![AML Best Run: ](images/3.PNG) 
![AML Best Run Completed: ](images/4.PNG) 

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search

Azure Hyperdrive package helps to automate choosing hyperparameter, which are adjustable parameters you choose for model training that guides the training process.

The model used is Logistic Regression as it utilizes a sigmoid function and works best on binary classification problems. It is implemented with scikit-learn.
The hyperparameters tuned are the inverse regularization strength -C ( Smaller values of C cause stronger regularization) and maximum iterations(max_iter) and a random sampling method over the search space.
The range for hyperparameters: 
Inverse regularization(C): (0.01, 0.1, 1, 10, 100, 1000, 10000)
Maximum iterations: (25,90,150,200)
A primary metric "Accuracy" is specified, which must be maximized to optimize the hyperparameter tuning experiment.
The parameters sampler chosen is Random Sampling, in which algorithm parameter values can be chosen from a set of discrete values or a distribution over a continuous range. Random sampling supports early termination of low-performance runs. We can users do an initial search with random sampling and then refine the search space to improve results.
Early stopping policy: I used a BanditPolicy with evaluation_interval of 5 and slack_factor of 0.1. Since early termination policies cause the poorly performing experiment runs to be cancelled so any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated. This means that after every 5 intervals, any run with its accuracy less than the best performing run's accuracy minus slack_factor 0.1 will be terminated. This saves us computational time since low-performance runs will be terminated.

### HuperDrive Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

The hyperdrive model model gave an accuracy of 81.6% with the following hyperparameters: C = 1000 and maximum iterations of 20. It performed better than AutoML’s best run so it will be deployed. The model could be improved by increasing the maximum number of iterations from 20.

![Hyperdrive Run Details:](images/6.PNG)
![Hyperdrive Top Runs:](images/7.PNG)
![Hyperdrive Run Primary Metric:](images/8.PNG)
![Hyperdrive parameters:](images/9.PNG)
![Hyperdrive Best Run and Parameters are given below:](images/10.PNG)
![Hyperdrive Best Run Details:](images/11.PNG)





## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.
 
The best model from hyperdrive run is first registered and then deployed locally. To deploy locally, the code is modified using LocalWebservice.deploy_configuration() to create a deployment configuration. Then we use Model.deploy() to deploy the ![service.](images/12.PNG)





![Here the ACI Webservice has been successfully deployed and it can be seen that the service state is healthy.](images/13.PNG)



 
 
Query Model Endpoint: For querying the endpoint, service.run() method is used. First,store the scoring uri,, then we create the header with key "Content-Type" and value "application/json" and then we create the sample input and post to the requests. Here is the sample input: 
data={"data":[{'SeniorCitizen':0, 'Partner': True, 'Dependents':False, 'PhoneService':False,'PaperlessBilling':True, 'MonthlyCharges':29.85, 'TotalCharges':29.85,'InternetService_DSL':1, 'InternetService_Fiber optic':0,'InternetService_No':0, 'OnlineSecurity_No':1,'OnlineSecurity_No internet service':1, 'OnlineSecurity_Yes':1,'OnlineBackup_No':0, 'OnlineBackup_No internet service':0,'OnlineBackup_Yes':1, 'DeviceProtection_No':1 ,'DeviceProtection_No internet service':0, 'DeviceProtection_Yes':0,'TechSupport_No':0, 'TechSupport_No internet service': 0, 'TechSupport_Yes':0,'StreamingTV_No':0, 'StreamingTV_No internet service':0, 'StreamingTV_Yes':0,'StreamingMovies_No':1, 'StreamingMovies_No internet service':0,'StreamingMovies_Yes':0, 'Contract_Month-to-month':1 , 'Contract_One year':0,'Contract_Two year':0}]}
The response status code is 200 which indicates that the request has succeeded. Then we use the service.run method to print the predictions. The model gives “False” which means that the customer has not ![churned.](images/14.PNG)
Endpoint status can be seen ![below.](images/15.PNG)

REST Endpoint is visible ![here](images/16.PNG) 



## Screen Recording [Link](https://drive.google.com/file/d/1ESHHbGsoX0LrqtUuhcB1GBS_Ncj-z-4_/view?usp=sharing)


## Standout Suggestions
Application Insights have been ![enabled.](images/17.PNG)


