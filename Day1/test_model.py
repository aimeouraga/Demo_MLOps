import pytest
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from dataset import X, y  

def test_logistic_regression_accuracy():

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LogisticRegression(max_iter=400)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    
    # Assert that accuracy is greater than a threshold
    assert accuracy >= 0.7, f"Accuracy {accuracy} is below the acceptable threshold."
