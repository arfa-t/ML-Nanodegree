import numpy as np
import pandas as pd
import os
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory
from azureml.core import Workspace, Dataset, Experiment
from azureml.core.dataset import Dataset
from sklearn.metrics import average_precision_score
 


def clean_data(data):
    data=data.to_pandas_dataframe()
    # Converting Total Charges to a numerical data type.
    data['TotalCharges']=data['TotalCharges'].apply(pd.to_numeric, errors='coerce') 
    
    #Removing missing values 
    data.dropna(inplace = True)
    #Remove customer IDs from the data set
    df2 = data.iloc[:,1:]
    #Converting the predictor variable in a binary numeric variable
    df2['Churn'].replace(to_replace='Yes', value=1, inplace=True)
    df2['Churn'].replace(to_replace='No',  value=0, inplace=True)

    
    #dropping unnecessary columns
    unwantedcolumnlist=["gender","MultipleLines","PaymentMethod","tenure"]


    df=df2.copy()
    df = df.drop(unwantedcolumnlist, axis=1)

    #Let's convert all the categorical variables into dummy variables
    df_dummies = pd.get_dummies(df)
    y = df_dummies[['Churn']]
    x = df_dummies.drop(columns = ['Churn'])
   
    return x,y
ds = TabularDatasetFactory.from_delimited_files(path="https://raw.githubusercontent.com/arfa-t/ML-Nanodegree/main/Capstone%20Project/churn.csv")

x,y =clean_data(ds) 
#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,shuffle=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC
import argparse
import joblib
from sklearn.metrics import accuracy_score


def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")


    args = parser.parse_args()

    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)


    # Make predictions
    preds = model.predict(x_test)
    
   
 
    
    accuracy = accuracy_score(y_test, preds)
    print ("accuracy : ", accuracy)
    run.log("accuracy", np.float(accuracy))
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=model, filename='outputs/hypermodel.pkl')
    

if __name__ == '__main__':
    main()

