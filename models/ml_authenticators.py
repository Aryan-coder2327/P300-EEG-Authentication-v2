import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class MLAuthenticator:
    
    def __init__(self, method='lda', random_state=42):
        self.method = method
        self.random_state = random_state
        
        if method == 'lda':
            self.model = LinearDiscriminantAnalysis()

        elif method == 'svm':
            self.model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True)

        elif method == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=random_state
            )

        else:
            raise ValueError("Method must be 'lda', 'svm', or 'rf'")
    
    
    def train(self, X_train, y_train, verbose=False):
        self.model.fit(X_train, y_train)

        if verbose:
            train_acc = self.model.score(X_train, y_train)
            print(f"Training accuracy: {train_acc:.2%}")


    def authenticate(self, X_test, y_test, verbose=False):

        predictions = self.model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        if hasattr(self.model, 'predict_proba'):
            scores = self.model.predict_proba(X_test).max(axis=1)
        else:
            scores = self.model.decision_function(X_test)

        if verbose:
            print(f"Test accuracy: {accuracy:.2%}")

        return predictions, scores, accuracy


    def get_scores_for_biometrics(self, X_test, y_test, target_subject):

        if hasattr(self.model, 'predict_proba'):
            scores = self.model.predict_proba(X_test)
            target_class_idx = target_subject

            if target_class_idx < scores.shape[1]:
                all_scores = scores[:, target_class_idx]
            else:
                all_scores = scores.max(axis=1)

        else:
            all_scores = self.model.decision_function(X_test)


        genuine_mask = y_test == target_subject
        impostor_mask = y_test != target_subject

        genuine_scores = all_scores[genuine_mask]
        impostor_scores = all_scores[impostor_mask]

        return genuine_scores, impostor_scores


    def get_model_info(self):
        return {
            'method': self.method,
            'model': self.model
        }
