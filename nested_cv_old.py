import numpy as np
import optuna
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


class NestedCVOptimizer:
    def __init__(self, X, y, groups, n_outer_folds=5, n_inner_folds=3,
                 n_trials=100, random_state=42):
        """
        Nested cross-validation optimizer for binary classification
        with optional imbalance handling and early stopping for XGBoost.
        """
        self.X = X
        self.y = y
        self.groups = groups
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.n_trials = n_trials
        self.random_state = random_state

        # Global label encoder for consistency across folds
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)

        # Cross-validation splitters
        self.outer_cv = GroupKFold(n_splits=n_outer_folds)
        self.inner_cv = GroupKFold(n_splits=n_inner_folds)

        # Tracking results
        self.outer_scores = []
        self.best_configs = []

    def define_search_space(self, trial):
        """Define model hyperparameter search space."""
        model_name = trial.suggest_categorical('model', ['rf', 'xgb'])

        if model_name == 'rf':
            params = {
                'model': 'rf',
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        else:  # XGBoost
            params = {
                'model': 'xgb',
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
                'early_stopping_rounds': trial.suggest_int('early_stopping_rounds', 10, 100)
            }
        return params

    
    def create_model(self, params):
        """Instantiate the model with given parameters."""
        if params['model'] == 'rf':
            return RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                bootstrap=params['bootstrap'],
                random_state=self.random_state,
                n_jobs=-1
            )
        elif params['model'] == 'xgb':
            return XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                min_child_weight=params['min_child_weight'],
                gamma=params['gamma'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss',
                early_stopping_rounds=params['early_stopping_rounds'],
                verbosity=0,
                use_label_encoder=False
            )

    def calculate_metrics(self, y_true, y_pred):
        """Compute binary classification metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }

    def inner_cv_objective(self, trial, X_train_outer, y_train_outer, groups_train_outer):
        """Inner CV loop for hyperparameter tuning."""
        params = self.define_search_space(trial)
        fold_scores = []

        for train_idx, val_idx in self.inner_cv.split(X_train_outer, y_train_outer, groups_train_outer):
            X_train_inner = X_train_outer.iloc[train_idx]
            X_val_inner = X_train_outer.iloc[val_idx]
            y_train_inner = y_train_outer.iloc[train_idx]
            y_val_inner = y_train_outer.iloc[val_idx]

            model = self.create_model(params)

            if params['model'] == 'xgb':
                model.fit(
                    X_train_balanced,
                    y_train_balanced,
                    eval_set=[(X_val_inner, y_val_inner)],
                    verbose=False
                )
            else:
                model.fit(X_train_balanced, y_train_balanced)

            y_pred = model.predict(X_val_inner)
            score = f1_score(y_val_inner, y_pred, zero_division=0)
            fold_scores.append(score)

        return np.mean(fold_scores)

    def run_nested_cv(self):
        """Run the full nested CV optimization."""
        print("Starting Nested Cross-Validation...")

        for fold_idx, (train_idx, test_idx) in enumerate(self.outer_cv.split(self.X, self.y_encoded, self.groups)):
            print(f"\n=== Outer Fold {fold_idx + 1}/{self.n_outer_folds} ===")

            X_train_outer = self.X.iloc[train_idx]
            X_test_outer = self.X.iloc[test_idx]
            y_train_outer = pd.Series(self.y_encoded[train_idx])
            y_test_outer = pd.Series(self.y_encoded[test_idx])
            groups_train_outer = self.groups.iloc[train_idx]

            # Inner CV hyperparameter tuning
            study = optuna.create_study(
                direction='maximize',
                sampler=optuna.samplers.TPESampler(seed=self.random_state),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
            )
            study.optimize(
                lambda trial: self.inner_cv_objective(trial, X_train_outer, y_train_outer, groups_train_outer),
                n_trials=self.n_trials,
                show_progress_bar=True
            )

            best_params = study.best_params
            print(f"Best parameters: {best_params}")

            # Retrain final model on full outer train set
            X_train_balanced, y_train_balanced = self.apply_imbalance_handling(
                X_train_outer, y_train_outer, best_params['imbalance_technique']
            )
            final_model = self.create_model(best_params)

            if best_params['model'] == 'xgb':
                final_model.fit(
                    X_train_balanced,
                    y_train_balanced,
                    eval_set=[(X_test_outer, y_test_outer)],
                    verbose=False
                )
            else:
                final_model.fit(X_train_balanced, y_train_balanced)

            # Evaluate on outer test set
            y_pred = final_model.predict(X_test_outer)
            fold_metrics = self.calculate_metrics(y_test_outer, y_pred)

            print(f"Outer fold metrics: {fold_metrics}")
            self.outer_scores.append(fold_metrics)
            self.best_configs.append(best_params)

        return self._summarize_results()

    def _summarize_results(self):
        """Summarize outer CV results."""
        results_df = pd.DataFrame(self.outer_scores)
        print("\nSummary:")
        print(results_df.mean(), "±", results_df.std())
        return {
            'mean_scores': results_df.mean().to_dict(),
            'std_scores': results_df.std().to_dict(),
            'individual_scores': self.outer_scores,
            'best_configs': self.best_configs
        }