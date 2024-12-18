import pandas as pd
import numpy as np
from django.shortcuts import render
from .forms import UploadFileForm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from pyswarm import pso
from skopt import BayesSearchCV

def upload_file(request):
    if request.method == 'POST':
        form = UploadFileForm(request.POST, request.FILES)
        if form.is_valid():
            file = request.FILES['file']
            df = pd.read_csv(file)

            # Data preprocessing
            X = df.drop(columns=['label'])
            y = df['label']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            non_numeric_columns = X_train.select_dtypes(exclude=[np.number]).columns
            X_train = X_train.drop(non_numeric_columns, axis=1)
            X_test = X_test.drop(non_numeric_columns, axis=1)
            X_train.columns = X_train.columns.str.strip().str.lower()
            X_test.columns = X_test.columns.str.strip().str.lower()

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # PSO for feature selection
            def objective_function(features):
                selected_features = [i for i in range(len(features)) if features[i] > 0.5]
                if len(selected_features) == 0:
                    return 1
                X_train_selected = X_train_scaled[:, selected_features]
                model = RandomForestClassifier()
                scores = cross_val_score(model, X_train_selected, y_train, cv=3, scoring='accuracy')
                return -scores.mean()

            num_features = X_train_scaled.shape[1]
            lower_bounds = [0] * num_features
            upper_bounds = [1] * num_features
            best_features, _ = pso(objective_function, lower_bounds, upper_bounds, swarmsize=15, maxiter=5)

            selected_features = [i for i in range(len(best_features)) if best_features[i] > 0.5]
            X_train_selected = X_train_scaled[:, selected_features]
            X_test_selected = X_test_scaled[:, selected_features]

            # Bayesian Optimization for hyperparameter tuning
            param_space = {
                'n_estimators': (10, 200),
                'max_depth': (3, 20),
                'min_samples_split': (2, 10),
                'min_samples_leaf': (1, 4)
            }
            bayes_search = BayesSearchCV(
                estimator=RandomForestClassifier(),
                search_spaces=param_space,
                n_iter=32,
                cv=5,
                n_jobs=-1,
                random_state=42
            )
            bayes_search.fit(X_train_selected, y_train)
            best_model = bayes_search.best_estimator_
            best_model.fit(X_train_selected, y_train)
            y_pred = best_model.predict(X_test_selected)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            return render(request, 'result.html', {
                'accuracy': accuracy * 100,
                'report': report
            })
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})