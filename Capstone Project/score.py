import json
import numpy as np
import os
import os
import pickle
import pandas as pd
from azureml.core import Model
#from sklearn.externals 
import joblib

data={"data":[{'SeniorCitizen':0, 'Partner': True, 'Dependents':False, 'PhoneService':False,
       'PaperlessBilling':True, 'MonthlyCharges':29.85, 'TotalCharges':29.85,
       'InternetService_DSL':1, 'InternetService_Fiber optic':0,
       'InternetService_No':0, 'OnlineSecurity_No':1,
       'OnlineSecurity_No internet service':0, 'OnlineSecurity_Yes':0,
       'OnlineBackup_No':0, 'OnlineBackup_No internet service':0,
       'OnlineBackup_Yes':1, 'DeviceProtection_No':1 ,
       'DeviceProtection_No internet service':0, 'DeviceProtection_Yes':0,
       'TechSupport_No':1, 'TechSupport_No internet service': 0, 'TechSupport_Yes':0,
       'StreamingTV_No':1, 'StreamingTV_No internet service':0, 'StreamingTV_Yes':0,
       'StreamingMovies_No':1, 'StreamingMovies_No internet service':0,
       'StreamingMovies_Yes':0, 'Contract_Month-to-month':1 , 'Contract_One year':0,
       'Contract_Two year':0}]}

def init():
    global model
    print("INIT")
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'hypermodel.pkl')
   #model_path = Model.get_model_path('hypermodel')
    model = joblib.load(model_path)

def run(data):
    try:
        print("Run")
        inp=json.loads(data)
        result = model.predict(pd.DataFrame(inp['data']))
        # You can return any data type, as long as it is JSON serializable.
        print("Successful predictions")
        return result.tolist()
    except Exception as e:
        error = str(e)
        return error