import unittest  # For unit testing
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add parent directory to path

from src.train import accuracy_score, load_iris, RandomForestClassifier, train_test_split

class TestModel(unittest.TestCase):
    def test_accuracy(self):
        # Load data and split into train/test sets
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
        
        # Train Random Forest model and make predictions
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        # Assert accuracy is above 0.5
        self.assertGreater(accuracy_score(y_test, predictions), 0.5)


if __name__ == "__main__":
    unittest.main()

