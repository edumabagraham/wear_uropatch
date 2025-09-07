import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
import pandas as pd
import warnings
from typing import Dict, List, Any, Tuple
warnings.filterwarnings('ignore')


class NestedCVOptimizer:
    def __init__(self, X, y, groups, sampling_tech, positive_class="void", n_outer_folds=5, n_inner_folds=3,
                 n_trials=50, random_state=42):
        """
        Nested cross-validation optimizer for binary classification
        with RandomForest, XGBoost, and DecisionTree.
        
        Each model gets its own hyperparameter optimization in each outer fold.
        
        Parameters:
        -----------
        positive_class : str
            The label that should be treated as the positive class (default: "void")
        """
        self.X = X
        self.y = y
        self.groups = groups
        self.sampling_tech = sampling_tech
        self.positive_class = positive_class
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.n_trials = n_trials
        self.random_state = random_state

        # Global label encoder for consistency across folds
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Identify the encoded value for the positive class
        self.positive_class_encoded = self.label_encoder.transform([self.positive_class])[0] # type: ignore
        print(f"Positive class '{self.positive_class}' is encoded as: {self.positive_class_encoded}")
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}") # type: ignore

        # Cross-validation splitters
        self.outer_cv = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
        self.inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)

        
        self.models = ['rf', 'xgb', 'dt']
        self.outer_scores = {model: [] for model in self.models}
        self.best_params_per_fold = {model: [] for model in self.models}
        self.optimization_histories = {model: [] for model in self.models}

    def get_model_search_space(self, model_name: str, trial) -> Dict[str, Any]:
        """Define model-specific hyperparameter search space."""
        if model_name == 'rf':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False])
            }
        elif model_name == 'xgb':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 2)
            }
        elif model_name == 'dt':
            return {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'splitter': trial.suggest_categorical('splitter', ['best', 'random'])
            }
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def create_model(self, model_name: str, params: Dict[str, Any]):
        """Instantiate the model with given parameters."""
        if model_name == 'rf':
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
        elif model_name == 'xgb':
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
                verbosity=0,
                use_label_encoder=False
            )
        elif model_name == 'dt':
            return DecisionTreeClassifier(
                criterion=params['criterion'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                splitter=params['splitter'],
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")

    def calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate comprehensive binary classification metrics."""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                
                # Binary classification metrics for positive class (void)
                'precision_positive': precision_score(y_true, y_pred, pos_label=self.positive_class_encoded, zero_division=0),
                'recall_positive': recall_score(y_true, y_pred, pos_label=self.positive_class_encoded, zero_division=0),
                'f1_positive': f1_score(y_true, y_pred, pos_label=self.positive_class_encoded, zero_division=0),
                
                # Binary classification metrics for negative class (non-void) for completeness
                'precision_negative': precision_score(y_true, y_pred, pos_label=1-self.positive_class_encoded, zero_division=0),
                'recall_negative': recall_score(y_true, y_pred, pos_label=1-self.positive_class_encoded, zero_division=0),
                'f1_negative': f1_score(y_true, y_pred, pos_label=1-self.positive_class_encoded, zero_division=0)
            }
            
            return metrics
        except Exception as e:
            print(f"Warning: Error calculating metrics: {e}")
            return {
                'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                'precision_weighted': 0.0, 'recall_weighted': 0.0, 'f1_weighted': 0.0,
                'precision_positive': 0.0, 'recall_positive': 0.0, 'f1_positive': 0.0,
                'precision_negative': 0.0, 'recall_negative': 0.0, 'f1_negative': 0.0
            }
            
            
    # This is where the inner hyperparameter tuning happens

    def optimize_model(self, model_name: str, X_train_outer, y_train_outer, groups_train_outer) -> Tuple[Dict, optuna.Study]:
        """Optimize hyperparameters for a specific model using inner CV."""
        def objective(trial):
            params = self.get_model_search_space(model_name, trial)
            fold_scores = []

            for train_idx, val_idx in self.inner_cv.split(X_train_outer, y_train_outer, groups_train_outer):
                try:
                    X_train_inner = X_train_outer.iloc[train_idx]
                    X_val_inner = X_train_outer.iloc[val_idx]
                    y_train_inner = y_train_outer.iloc[train_idx]
                    y_val_inner = y_train_outer.iloc[val_idx]
                    
                    if self.sampling_tech == 'oversample':
                        sampler = SMOTE(random_state=self.random_state)                
                    elif self.sampling_tech == 'undersample':
                        sampler = TomekLinks()                
                    elif self.sampling_tech == 'smotetomek':
                        sampler = SMOTETomek(random_state=self.random_state)
                        
                    try:
                        resampled = sampler.fit_resample(X_train_inner, y_train_inner)
                        if isinstance(resampled, tuple) and len(resampled) == 3:
                            X_train_inner_resampled, y_train_inner_resampled, _ = resampled
                        else:
                            X_train_inner_resampled, y_train_inner_resampled = resampled
                        X_train_inner_resampled = pd.DataFrame(X_train_inner_resampled, columns=X_train_inner.columns)
                        y_train_inner_resampled = pd.Series(np.ravel(y_train_inner_resampled))
                    except Exception as e:
                        print(f"Sampling failed for {self.sampling_tech}: {e}")
                        continue

                    model = self.create_model(model_name, params)

                    if model_name == 'xgb':
                        model.fit(
                            X_train_inner_resampled, y_train_inner_resampled,
                            eval_set=[(X_val_inner, y_val_inner)],
                            verbose=False
                        )
                    else:
                        model.fit(X_train_inner_resampled, y_train_inner_resampled)

                    y_pred = model.predict(X_val_inner)
                    # For binary classification, we can optimize on F1 score of positive class (void)
                    score = f1_score(y_val_inner, y_pred, pos_label=self.positive_class_encoded, zero_division=0)
                    fold_scores.append(score)
                    
                except Exception as e:
                    print(f"Warning: Error in inner fold for {model_name}: {e}")
                    fold_scores.append(0.0)

            return np.mean(fold_scores) if fold_scores else 0.0

        # Create study for this model
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params, study

    def run_nested_cv(self) -> Dict[str, Any]:
        """Run the complete nested CV optimization for all models."""
        print("Starting Binary Nested Cross-Validation with separate optimization for each model...")
        print(f"Positive class: '{self.positive_class}' (encoded as {self.positive_class_encoded})")

        for fold_idx, (train_idx, test_idx) in enumerate(self.outer_cv.split(self.X, self.y_encoded, self.groups)):
            print(f"\n{'='*60}")
            print(f"OUTER FOLD {fold_idx + 1}/{self.n_outer_folds}")
            print(f"{'='*60}")

            # Prepare outer fold data
            X_train_outer = self.X.iloc[train_idx]
            X_test_outer = self.X.iloc[test_idx]
            y_train_outer = pd.Series(self.y_encoded, index=self.X.index).iloc[train_idx]
            y_test_outer = pd.Series(self.y_encoded, index=self.X.index).iloc[test_idx]
            groups_train_outer = self.groups.iloc[train_idx] if hasattr(self.groups, 'iloc') else self.groups[train_idx]

            print(f"Train size: {len(X_train_outer)}, Test size: {len(X_test_outer)}")
            
            # Show class distribution with original labels
            test_classes, test_counts = np.unique(y_test_outer, return_counts=True)
            test_class_names = self.label_encoder.inverse_transform(test_classes)
            print(f"Class distribution in test set: {dict(zip(test_class_names, test_counts))}")
            
            
            
            # Optimize each model separately
            for model_name in self.models:
                print(f"\n--- Optimizing {model_name.upper()} ---")
                
                try:
                    best_params, study = self.optimize_model(model_name, X_train_outer, y_train_outer, groups_train_outer)
                    print(f"Best {model_name} params: {best_params}")
                    print(f"Best {model_name} F1 score (positive class): {study.best_value:.4f}")

                    # Store optimization results
                    self.best_params_per_fold[model_name].append(best_params)
                    self.optimization_histories[model_name].append({
                        'best_value': study.best_value,
                        'n_trials': len(study.trials)
                    })
                    
                    # Apply same sampling to outer training set for final model
                    if self.sampling_tech == 'oversample':
                        sampler = SMOTE(random_state=self.random_state)                
                    elif self.sampling_tech == 'undersample':
                        sampler = TomekLinks()                
                    elif self.sampling_tech == 'smotetomek':
                        sampler = SMOTETomek(random_state=self.random_state)
                        
                    try:
                        resampled = sampler.fit_resample(X_train_outer, y_train_outer)
                        if isinstance(resampled, tuple) and len(resampled) == 3:
                            X_train_outer_resampled, y_train_outer_resampled, _ = resampled
                        else:
                            X_train_outer_resampled, y_train_outer_resampled = resampled
                        X_train_outer_resampled = pd.DataFrame(X_train_outer_resampled, columns=X_train_outer.columns)
                        y_train_outer_resampled = pd.Series(np.ravel(y_train_outer_resampled))
                        
                        print(f"Applied {self.sampling_tech} to outer training set: {len(X_train_outer)} -> {len(X_train_outer_resampled)} samples")
                    
                    except Exception as e:
                        print(f"Sampling failed for {self.sampling_tech}: {e}")
                        X_train_outer_resampled, y_train_outer_resampled = X_train_outer, y_train_outer
                        # continue
                        
                        
                    # Train final model on full resampled outer training set
                    final_model = self.create_model(model_name, best_params)
                    if model_name == 'xgb':
                        final_model.fit(X_train_outer_resampled, y_train_outer_resampled, verbose=False)
                    else:
                        final_model.fit(X_train_outer_resampled, y_train_outer_resampled)

                    # Evaluate on outer test set
                    y_pred = final_model.predict(X_test_outer)
                    fold_metrics = self.calculate_metrics(y_test_outer, y_pred)
                    self.outer_scores[model_name].append(fold_metrics)

                    print(f"{model_name} test accuracy: {fold_metrics['accuracy']:.4f}")
                    print(f"{model_name} test F1 (positive): {fold_metrics['f1_positive']:.4f}")

                except Exception as e:
                    print(f"Error optimizing {model_name}: {e}")
                    # Store default results for failed optimization
                    self.best_params_per_fold[model_name].append({})
                    self.optimization_histories[model_name].append({'best_value': 0.0, 'n_trials': 0})
                    default_metrics = {
                        'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                        'precision_weighted': 0.0, 'recall_weighted': 0.0, 'f1_weighted': 0.0,
                        'precision_positive': 0.0, 'recall_positive': 0.0, 'f1_positive': 0.0,
                        'precision_negative': 0.0, 'recall_negative': 0.0, 'f1_negative': 0.0
                    }
                    self.outer_scores[model_name].append(default_metrics)

        return self._summarize_results()

    def _summarize_results(self) -> Dict[str, Any]:
        """Summarize and display results."""
        print(f"\n{'='*80}")
        print("BINARY NESTED CROSS-VALIDATION RESULTS SUMMARY")
        print(f"Positive class: '{self.positive_class}'")
        print(f"{'='*80}")

        summaries = {}
        
        for model_name in self.models:
            if self.outer_scores[model_name]:
                scores_df = pd.DataFrame(self.outer_scores[model_name])
                
                summaries[model_name] = {
                    'mean_scores': scores_df.mean().to_dict(),
                    'std_scores': scores_df.std().to_dict(),
                    'all_fold_scores': self.outer_scores[model_name],
                    'best_params_per_fold': self.best_params_per_fold[model_name],
                    'optimization_history': self.optimization_histories[model_name]
                }
                
                print(f"\n{model_name.upper()} Results:")
                print("-" * 40)
                # Show key binary classification metrics
                key_metrics = ['accuracy', 'f1_positive', 'precision_positive', 'recall_positive', 'f1_macro']
                for metric in key_metrics:
                    if metric in scores_df.columns:
                        mean_val = scores_df[metric].mean()
                        std_val = scores_df[metric].std()
                        print(f"{metric:18}: {mean_val:.4f} ± {std_val:.4f}")
                
                print(f"Individual fold accuracies: {[f'{score:.4f}' for score in scores_df['accuracy'].tolist()]}")
                print(f"Individual fold F1 (pos): {[f'{score:.4f}' for score in scores_df['f1_positive'].tolist()]}")

        # Find best model based on F1 score of positive class
        if summaries:
            best_model = max(summaries.keys(), 
                        key=lambda x: summaries[x]['mean_scores'].get('f1_positive', 0))
            best_f1_positive = summaries[best_model]['mean_scores']['f1_positive']
            best_accuracy = summaries[best_model]['mean_scores']['accuracy']
            
            print(f"\n{'='*80}")
            print(f"BEST MODEL: {best_model.upper()}")
            print(f"F1 Score (positive class): {best_f1_positive:.4f}")
            print(f"Accuracy: {best_accuracy:.4f}")
            print(f"{'='*80}")
            
            summaries['best_model'] = best_model
            summaries['best_f1_positive'] = best_f1_positive
            summaries['best_accuracy'] = best_accuracy

        return summaries

    def get_fold_results(self) -> Dict[str, List[float]]:
        """
        Get the accuracy scores for each model across all outer folds.
        """
        fold_accuracies = {}
        for model_name in self.models:
            accuracies = [fold_scores['accuracy'] for fold_scores in self.outer_scores[model_name]]
            fold_accuracies[model_name] = accuracies
        
        return fold_accuracies
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with fold results including binary classification metrics
        for the positive class ('void').
        """
        results_data = []
        
        # Add fold results
        for model_name in self.models:
            for fold_idx in range(self.n_outer_folds):
                # Get all metrics for this fold
                fold_metrics = self.outer_scores[model_name][fold_idx] if fold_idx < len(self.outer_scores[model_name]) else {}
                
                # Get best parameters for this fold
                best_params = self.best_params_per_fold[model_name][fold_idx] if (
                    self.best_params_per_fold[model_name] and fold_idx < len(self.best_params_per_fold[model_name])
                ) else {}
                
                # Create base row with all metrics
                row = {
                    'Model': model_name.upper(),
                    'Fold': fold_idx + 1,
                    'Accuracy': round(fold_metrics.get('accuracy', 0.0), 4),
                    'Precision_Macro': round(fold_metrics.get('precision_macro', 0.0), 4),
                    'Recall_Macro': round(fold_metrics.get('recall_macro', 0.0), 4),
                    'F1_Macro': round(fold_metrics.get('f1_macro', 0.0), 4),
                    'Precision_Weighted': round(fold_metrics.get('precision_weighted', 0.0), 4),
                    'Recall_Weighted': round(fold_metrics.get('recall_weighted', 0.0), 4),
                    'F1_Weighted': round(fold_metrics.get('f1_weighted', 0.0), 4),
                    
                    # Binary classification metrics for positive class ('void')
                    'Precision_Positive': round(fold_metrics.get('precision_positive', 0.0), 4),
                    'Recall_Positive': round(fold_metrics.get('recall_positive', 0.0), 4),
                    'F1_Positive': round(fold_metrics.get('f1_positive', 0.0), 4),
                    
                    # Binary classification metrics for negative class ('non-void')
                    'Precision_Negative': round(fold_metrics.get('precision_negative', 0.0), 4),
                    'Recall_Negative': round(fold_metrics.get('recall_negative', 0.0), 4),
                    'F1_Negative': round(fold_metrics.get('f1_negative', 0.0), 4),
                    
                    'Best_Params': str(best_params)  # As string for easy viewing
                }
                
                # Add each hyperparameter as a separate column for easier analysis
                for param_name, param_value in best_params.items():
                    row[f'HP_{param_name}'] = param_value
                
                results_data.append(row)
        
        # Create DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Debug: Print column names to see what's actually in the DataFrame
        print("Columns in results_df:", list(results_df.columns))
        
        # Add summary rows
        summary_data = []
        # Only include metrics that actually exist in the DataFrame
        all_metric_columns = ['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 
                            'Precision_Weighted', 'Recall_Weighted', 'F1_Weighted',
                            'Precision_Positive', 'Recall_Positive', 'F1_Positive',
                            'Precision_Negative', 'Recall_Negative', 'F1_Negative']
        
        # Filter to only include columns that exist in the DataFrame
        metric_columns = [col for col in all_metric_columns if col in results_df.columns]
        print("Available metric columns:", metric_columns)
        
        for model_name in self.models:
            model_data = results_df[results_df['Model'] == model_name.upper()]
            
            if model_data.empty:
                print(f"Warning: No data found for model {model_name}")
                continue
            
            # Mean row
            mean_row = {
                'Model': model_name.upper(),
                'Fold': 'MEAN',
                'Best_Params': 'N/A'
            }
            for metric in metric_columns:
                if metric in model_data.columns:
                    mean_row[metric] = round(model_data[metric].mean(), 4)
                else:
                    mean_row[metric] = 0.0
            summary_data.append(mean_row)
            
            # Std row  
            std_row = {
                'Model': model_name.upper(),
                'Fold': 'STD',
                'Best_Params': 'N/A'
            }
            for metric in metric_columns:
                if metric in model_data.columns:
                    std_row[metric] = round(model_data[metric].std(), 4)
                else:
                    std_row[metric] = 0.0
            summary_data.append(std_row)
        
        # Combine all results
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            final_df = pd.concat([results_df, summary_df], ignore_index=True)
        else:
            final_df = results_df
        
        return final_df