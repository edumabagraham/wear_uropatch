import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings
from typing import Dict, List, Any, Tuple
warnings.filterwarnings('ignore')


class NestedCVOptimizer:
    def __init__(self, X, y, groups, n_outer_folds=5, n_inner_folds=3,
                 n_trials=50, random_state=42):
        """
        Improved nested cross-validation optimizer for multi-class classification
        with RandomForest, XGBoost, and DecisionTree.
        
        Each model gets its own hyperparameter optimization in each outer fold.
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
        self.outer_cv = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
        self.inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)

        # Results storage - now properly structured per model per fold
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
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),  # Removed 'log_loss' for compatibility
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
                eval_metric='mlogloss',
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
        """Calculate comprehensive classification metrics."""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # Add per-class metrics if we have the expected classes
            unique_classes = np.unique(y_true)
            if len(unique_classes) <= 3:  # Only add per-class if reasonable number of classes
                precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
                recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
                f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
                
                for i, class_val in enumerate(unique_classes):
                    if i < len(precision_per_class):
                        metrics[f'precision_class_{class_val}'] = precision_per_class[i]
                        metrics[f'recall_class_{class_val}'] = recall_per_class[i]
                        metrics[f'f1_class_{class_val}'] = f1_per_class[i]
            
            return metrics
        except Exception as e:
            print(f"Warning: Error calculating metrics: {e}")
            return {'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0}

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

                    model = self.create_model(model_name, params)

                    if model_name == 'xgb':
                        model.fit(
                            X_train_inner, y_train_inner,
                            eval_set=[(X_val_inner, y_val_inner)],
                            verbose=False
                        )
                    else:
                        model.fit(X_train_inner, y_train_inner)

                    y_pred = model.predict(X_val_inner)
                    score = accuracy_score(y_val_inner, y_pred)
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
        print("Starting Nested Cross-Validation with separate optimization for each model...")

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
            print(f"Class distribution in test set: {np.bincount(y_test_outer)}")

            # Optimize each model separately
            for model_name in self.models:
                print(f"\n--- Optimizing {model_name.upper()} ---")
                
                try:
                    best_params, study = self.optimize_model(model_name, X_train_outer, y_train_outer, groups_train_outer)
                    print(f"Best {model_name} params: {best_params}")
                    print(f"Best {model_name} CV score: {study.best_value:.4f}")

                    # Store optimization results
                    self.best_params_per_fold[model_name].append(best_params)
                    self.optimization_histories[model_name].append({
                        'best_value': study.best_value,
                        'n_trials': len(study.trials)
                    })

                    # Train final model on full outer training set
                    final_model = self.create_model(model_name, best_params)
                    if model_name == 'xgb':
                        final_model.fit(X_train_outer, y_train_outer, verbose=False)
                    else:
                        final_model.fit(X_train_outer, y_train_outer)

                    # Evaluate on outer test set
                    y_pred = final_model.predict(X_test_outer)
                    fold_metrics = self.calculate_metrics(y_test_outer, y_pred)
                    self.outer_scores[model_name].append(fold_metrics)

                    print(f"{model_name} test accuracy: {fold_metrics['accuracy']:.4f}")

                except Exception as e:
                    print(f"Error optimizing {model_name}: {e}")
                    # Store default results for failed optimization
                    self.best_params_per_fold[model_name].append({})
                    self.optimization_histories[model_name].append({'best_value': 0.0, 'n_trials': 0})
                    self.outer_scores[model_name].append(self.calculate_metrics([0], [0]))  # Dummy metrics

        return self._summarize_results()

    def _summarize_results(self) -> Dict[str, Any]:
        """Summarize and display results."""
        print(f"\n{'='*80}")
        print("NESTED CROSS-VALIDATION RESULTS SUMMARY")
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
                key_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
                for metric in key_metrics:
                    if metric in scores_df.columns:
                        mean_val = scores_df[metric].mean()
                        std_val = scores_df[metric].std()
                        print(f"{metric:15}: {mean_val:.4f} Â± {std_val:.4f}")
                
                print(f"Individual fold accuracies: {[f'{score:.4f}' for score in scores_df['accuracy'].tolist()]}")

        # Find best model
        if summaries:
            best_model = max(summaries.keys(), 
                           key=lambda x: summaries[x]['mean_scores'].get('accuracy', 0))
            best_accuracy = summaries[best_model]['mean_scores']['accuracy']
            
            print(f"\n{'='*80}")
            print(f"BEST MODEL: {best_model.upper()} (Accuracy: {best_accuracy:.4f})")
            print(f"{'='*80}")
            
            summaries['best_model'] = best_model
            summaries['best_accuracy'] = best_accuracy

        return summaries

    def get_fold_results(self) -> Dict[str, List[float]]:
        """
        Get the accuracy scores for each model across all outer folds.
        This is the main result you wanted to analyze.
        """
        fold_accuracies = {}
        for model_name in self.models:
            accuracies = [fold_scores['accuracy'] for fold_scores in self.outer_scores[model_name]]
            fold_accuracies[model_name] = accuracies
        
        return fold_accuracies
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with fold accuracies, macro metrics, and best hyperparameters.
        Returns a clean DataFrame ready for analysis.
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
                    'Best_Params': str(best_params)  # As string for easy viewing
                }
                
                # Add each hyperparameter as a separate column for easier analysis
                for param_name, param_value in best_params.items():
                    row[f'HP_{param_name}'] = param_value
                
                results_data.append(row)
        
        # Create DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Add summary rows
        summary_data = []
        metric_columns = ['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 
                         'Precision_Weighted', 'Recall_Weighted', 'F1_Weighted']
        
        for model_name in self.models:
            model_data = results_df[results_df['Model'] == model_name.upper()]
            
            # Mean row
            mean_row = {
                'Model': model_name.upper(),
                'Fold': 'MEAN',
                'Best_Params': 'N/A'
            }
            for metric in metric_columns:
                mean_row[metric] = round(model_data[metric].mean(), 4)
            summary_data.append(mean_row)
            
            # Std row  
            std_row = {
                'Model': model_name.upper(),
                'Fold': 'STD',
                'Best_Params': 'N/A'
            }
            for metric in metric_columns:
                std_row[metric] = round(model_data[metric].std(), 4)
            summary_data.append(std_row)
        
        # Combine all results
        summary_df = pd.DataFrame(summary_data)
        final_df = pd.concat([results_df, summary_df], ignore_index=True)
        
        return final_df