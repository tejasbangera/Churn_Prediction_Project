from sklearn.linear_model import LogisticRegression
import argparse
import os
import numpy as np
from sklearn.metrics import mean_squared_error
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

# TODO: Create TabularDataset using TabularDatasetFactory
# Data is located at:
# "https://raw.githubusercontent.com/tejasbangera/Udacity-Captstone-Project/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"

#ds = ### YOUR CODE HERE ###
file_url = "https://raw.githubusercontent.com/tejasbangera/Udacity-Captstone-Project/main/WA_Fn-UseC_-Telco-Customer-Churn.csv"
ds = TabularDatasetFactory.from_delimited_files(file_url)

# TODO: Split data into train and test sets.

### YOUR CODE HERE ###a

def clean_data(data):
    # Clean and one hot encode data
    x_df = data.to_pandas_dataframe().dropna()
    x_df['Churn']=x_df.Churn.apply(lambda s:1 if s==True else 0)
    x_df = pd.get_dummies(x_df, drop_first=True)
    y_df = x_df.pop("Churn")
    return x_df,y_df

def split_data(ds):
    x, y = clean_data(ds)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=1)
    return (x_train, x_test, y_train, y_test)

run = Run.get_context()

def main():
    # Add arguments to script
    (x_train, x_test, y_train, y_test) = split_data(ds)
    parser = argparse.ArgumentParser()

    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")

    args = parser.parse_args()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))

    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
