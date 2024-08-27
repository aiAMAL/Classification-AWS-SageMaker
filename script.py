
import numpy as np
import pandas as pd

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

import boto3
import joblib
import pathlib
from io import StringIO
import argparse


def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, 'model.joblib'))
    return clf

if __name__ = "__main__":
    print('[INFO] Extracting arguments')
    parser = argparse.ArgumentParser()

    # Hyperparameters sent by the client are passed as command-line arguments to the scripts.
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Data, model, and output directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))
    parser.add_argument("--train-file", type=str, default="Mobile_Price_Classification_Train_V1.csv")
    parser.add_argument("--test-file", type=str, default="Mobile_Price_Classification_Test_V1.csv")

    args, _ = parser.parse_known_args()

    print("SKLearn Version", skearn.__version__)
    print("Joblib Version", joblib.__version__)

    print('[INFO] Reading Data')
    train_df = pd.read_csv(os.path.join(args.train, args.train_file))
    test_df = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(train_df.columns)
    label = features.pop()

    print('Building training and testing datasets')
    X_train = train_df[features]
    y_train = train_df[label]
    X_test = test_df[features]    
    y_test = test_df[label]

    print('Column order: ', features)
    print('Label:', label)

    print('Data shape')
    print('Training Data shape: ')
    print(X_train.shape)
    print(y_train.shape)

    print('Test Data shape: ')
    print(X_test.shape)
    print(y_test.shape)
    print()
    
    print('Training RandomForest Model ...')
    model = RandomForestClassifier(n_estimaors=args.n_estimators , random_state=args.random_state, verbose=True)
    model.fit(X_train, y_train)
    print()

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print('Model persisted at', model_path)
    print()

    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)

    print()
    print('------ Metrics Results for Testing Data --------')
    print()
    print('Total Rows are: ', X_test.shape[0])
    print('[TESTING] Model Accuracy is :', test_acc)
    print('[TESTING] Testing Report: ')
    print(test_report)

    












