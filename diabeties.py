import mlflow
import mlflow.dvc
from dagshub import dagshub_logger
mlflow.set_tracking_uri('https://dagshub.com/Rajeshhugar/mlflow_diabetes.mlflow')

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

with dagshub_logger() as logger:
    mlflow.sklearn.autolog()

    # Your existing code here
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    # Create and train models.
    rf = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    predictions = rf.predict(X_test)
