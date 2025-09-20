import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                        roc_curve, auc, roc_auc_score, log_loss)
from sklearn.preprocessing import LabelEncoder, label_binarize
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Any, Tuple
from sklearn.metrics import confusion_matrix
import os
warnings.filterwarnings('ignore')

class ModifiedNestedCVOptimizer:
    """
    Nested cross-validation optimizer adapted for global normalization with pre-split data.
    
    This class adapts your original NestedCVOptimizer to work with pre-split data from 
    global normalization while maintaining the exact same output format.
    """
    
    def __init__(self, n_inner_folds=3, n_trials=50, random_state=42):
        """
        Initialize the optimizer for pre-split global normalization data.
        
        Parameters:
        -----------
        n_inner_folds : int
            Number of folds for inner cross-validation (hyperparameter optimization)
        n_trials : int
            Number of optimization trials for each model
        random_state : int
            Random state for reproducibility
        """
        self.n_inner_folds = n_inner_folds
        self.n_trials = n_trials
        self.random_state = random_state
        
        
        # Inner CV splitter
        self.inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
        
        # Models to evaluate
        self.models = ['rf', 'xgb', 'dt']
        
        # Results storage (per configuration)
        self.current_config_results = []
        self.all_configurations_results = {}
        
        # Current configuration context
        self.current_config = None
        self.current_n_features = None
    
    def get_model_search_space(self, model_name: str, trial) -> Dict[str, Any]:
        """Define model-specific hyperparameter search space with SelectKBest."""
        
        # Feature selection parameters
        feature_selection_k = trial.suggest_categorical('selector__k', 
                                                       [int(self.current_n_features * 0.1), 
                                                        int(self.current_n_features * 0.2),
                                                        int(self.current_n_features * 0.3),
                                                        int(self.current_n_features * 0.4),
                                                        int(self.current_n_features * 0.5),
                                                        int(self.current_n_features * 0.7),
                                                        int(self.current_n_features * 0.8),
                                                        int(self.current_n_features * 0.9),
                                                        'all'])
        
        score_func = trial.suggest_categorical('selector__score_func', ['f_classif'])
        
        params = {
            'selector__k': feature_selection_k,
            'selector__score_func': score_func
        }

        if model_name == 'rf':
            params.update({
                'clf__n_estimators': trial.suggest_int('clf__n_estimators', 50, 500),
                'clf__max_depth': trial.suggest_int('clf__max_depth', 3, 20),
                'clf__min_samples_split': trial.suggest_int('clf__min_samples_split', 2, 20),
                'clf__min_samples_leaf': trial.suggest_int('clf__min_samples_leaf', 1, 10),
                'clf__max_features': trial.suggest_categorical('clf__max_features', ['sqrt', 'log2', None]),
                'clf__bootstrap': trial.suggest_categorical('clf__bootstrap', [True, False]),
                'clf__class_weight': trial.suggest_categorical('clf__class_weight', [None, 'balanced'])
            })
        elif model_name == 'xgb':
            params.update({
                'clf__n_estimators': trial.suggest_int('clf__n_estimators', 50, 500),
                'clf__max_depth': trial.suggest_int('clf__max_depth', 3, 12),
                'clf__learning_rate': trial.suggest_float('clf__learning_rate', 0.01, 0.3, log=True),
                'clf__subsample': trial.suggest_float('clf__subsample', 0.6, 1.0),
                'clf__colsample_bytree': trial.suggest_float('clf__colsample_bytree', 0.6, 1.0),
                'clf__min_child_weight': trial.suggest_int('clf__min_child_weight', 1, 10),
                'clf__gamma': trial.suggest_float('clf__gamma', 0, 5),
                'clf__reg_alpha': trial.suggest_float('clf__reg_alpha', 0, 2),
                'clf__reg_lambda': trial.suggest_float('clf__reg_lambda', 0, 2)
            })
        elif model_name == 'dt':
            params.update({
                'clf__criterion': trial.suggest_categorical('clf__criterion', ['gini', 'entropy']),
                'clf__max_depth': trial.suggest_int('clf__max_depth', 1, 20),
                'clf__min_samples_split': trial.suggest_int('clf__min_samples_split', 2, 20),
                'clf__min_samples_leaf': trial.suggest_int('clf__min_samples_leaf', 1, 10),
                'clf__max_features': trial.suggest_categorical('clf__max_features', ['sqrt', 'log2', None]),
                'clf__splitter': trial.suggest_categorical('clf__splitter', ['best', 'random']),
                'clf__class_weight': trial.suggest_categorical('clf__class_weight', [None, 'balanced'])
            })
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return params

    def create_model(self, model_name: str, params: Dict[str, Any]):
        """Instantiate a Pipeline with SelectKBest and the model with given parameters."""
        
        # Extract classifier-specific parameters
        clf_params = {k.replace('clf__', ''): v for k, v in params.items() if k.startswith('clf__')}
        selector_params = {k.replace('selector__', ''): v for k, v in params.items() if k.startswith('selector__')}

        # Create the classifier instance
        if model_name == 'rf':
            clf = RandomForestClassifier(**clf_params, random_state=self.random_state, n_jobs=-1)
        elif model_name == 'xgb':
            clf = XGBClassifier(**clf_params, random_state=self.random_state, n_jobs=-1,
                                eval_metric='mlogloss', verbosity=0, use_label_encoder=False,
                                objective='multi:softprob')
        elif model_name == 'dt':
            clf = DecisionTreeClassifier(**clf_params, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Create the SelectKBest instance
        score_func_map = {
            'f_classif': f_classif
        }
        score_func = score_func_map[selector_params['score_func']]
        k = selector_params['k']
        
        selector = SelectKBest(score_func=score_func, k=k)

        # Create and return the pipeline
        pipeline = Pipeline([
            ('selector', selector),
            ('clf', clf)
        ])
        return pipeline
    
    def calculate_auc_scores(self, y_true, y_proba, fold_idx: int, model_name: str) -> Dict[str, Any]:
        """
        Calculate AUC scores for multi-class classification using one-vs-rest approach.
        
        Args:
            y_true: True labels (encoded)
            y_proba: Predicted probabilities for each class
            fold_idx: Current fold index
            model_name: Name of the model
            
        Returns:
            Dictionary containing AUC scores
        """
        auc_data = {
            'fold': fold_idx,
            'model': model_name,
            'macro_auc': 0.0,
            'micro_auc': 0.0
        }
        
        try:
            # Binarize the output for multi-class AUC
            y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
            
            # Handle case where test set doesn't contain all classes
            if y_true_bin.shape[1] < self.n_classes:
                # Pad with zeros for missing classes
                missing_classes = self.n_classes - y_true_bin.shape[1]
                padding = np.zeros((y_true_bin.shape[0], missing_classes))
                y_true_bin = np.hstack([y_true_bin, padding])
            
            # Calculate per-class AUC scores (one-vs-rest)
            per_class_aucs = []
            for i, class_name in enumerate(self.class_names):
                if i < y_proba.shape[1] and np.sum(y_true_bin[:, i]) > 0:
                    try:
                        auc_score = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
                        auc_data[f'auc_{class_name}'] = auc_score
                        per_class_aucs.append(auc_score)
                    except ValueError:
                        # Handle cases where only one class is present
                        auc_data[f'auc_{class_name}'] = 0.0
                else:
                    # Handle missing class in test set
                    auc_data[f'auc_{class_name}'] = 0.0
            
            # Calculate macro AUC (average of per-class AUCs)
            if per_class_aucs:
                auc_data['macro_auc'] = np.mean(per_class_aucs)
            
            # Calculate micro AUC (if possible)
            if y_proba.shape[1] >= self.n_classes and np.sum(y_true_bin) > 0:
                try:
                    auc_data['micro_auc'] = roc_auc_score(y_true_bin, y_proba, 
                                                        multi_class='ovr', average='micro')
                except ValueError:
                    auc_data['micro_auc'] = 0.0
            
        except Exception as e:
            print(f"Warning: Error calculating AUC scores for {model_name} fold {fold_idx}: {e}")
            # Return default values
            for class_name in self.class_names:
                auc_data[f'auc_{class_name}'] = 0.0
        
        return auc_data

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_proba) -> Dict[str, float]:
        """Calculate comprehensive binary classification metrics exactly like your original."""
        
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
            
            # Add per-class metrics with proper class names
            unique_classes = np.unique(y_true)
            if len(unique_classes) <= len(self.class_names):
                precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
                recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
                f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
                
                # Map encoded class indices to class names
                for encoded_idx, class_name in enumerate(self.class_names):
                    if encoded_idx < len(precision_per_class):
                        metrics[f'precision_{class_name}'] = precision_per_class[encoded_idx]
                        metrics[f'recall_{class_name}'] = recall_per_class[encoded_idx]
                        metrics[f'f1_{class_name}'] = f1_per_class[encoded_idx]
                    else:
                        # Handle cases where a class might not be present in this fold
                        metrics[f'precision_{class_name}'] = 0.0
                        metrics[f'recall_{class_name}'] = 0.0
                        metrics[f'f1_{class_name}'] = 0.0
            
            return metrics
        except Exception as e:
            print(f"Warning: Error calculating metrics: {e}")
            # Return default metrics including per-class defaults
            default_metrics = {
                'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                'precision_weighted': 0.0, 'recall_weighted': 0.0, 'f1_weighted': 0.0
            }
            for class_name in self.class_names:
                default_metrics[f'precision_{class_name}'] = 0.0
                default_metrics[f'recall_{class_name}'] = 0.0
                default_metrics[f'f1_{class_name}'] = 0.0
            return default_metrics

    def optimize_model(self, model_name: str, X_train, y_train, groups_train) -> Tuple[Dict, float]:
        """Optimize hyperparameters for a specific model using inner CV."""
        
        def objective(trial):
            params = self.get_model_search_space(model_name, trial)
            fold_scores = []

            for train_idx, val_idx in self.inner_cv.split(X_train, y_train, groups_train):
                try:
                    X_train_inner = X_train.iloc[train_idx]
                    X_val_inner = X_train.iloc[val_idx]
                    y_train_inner = y_train[train_idx]
                    y_val_inner = y_train[val_idx]

                    model = self.create_model(model_name, params)
                    model.fit(X_train_inner, y_train_inner)

                    # Get predicted probabilities for the validation set
                    y_proba = model.predict_proba(X_val_inner)

                    # Calculate negative log loss (maximize this = minimize log loss)
                    score = -log_loss(y_val_inner, y_proba)
                    fold_scores.append(score)
                    
                except Exception as e:
                    print(f"Warning: Error in inner fold for {model_name}: {e}")
                    # Use Optuna's pruning mechanism instead of arbitrary score
                    raise optuna.TrialPruned()

            return np.mean(fold_scores) if fold_scores else -np.inf

        # Create study for this model with proper seeding
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                seed=self.random_state,
                n_startup_trials=10,
                n_ei_candidates=24
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=10,
                n_warmup_steps=5,
                n_min_trials=1
            )
        )
        
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        
        return study.best_params, study.best_value

    def evaluate_single_fold(self, fold_data: Dict) -> List[Dict]:
        """Evaluate all models on a single fold's pre-split data."""
        
        fold_id = fold_data['fold']
        train_features = fold_data['train_features']
        test_features = fold_data['test_features']
        
        print(f"\n  OUTER FOLD {fold_id}")
        print(f"  Train: {train_features.shape}, Test: {test_features.shape}")
        
        # Prepare data
        X_train = train_features.drop(columns=['label', 'experiment_id'])
        y_train = train_features['label']
        groups_train = train_features['experiment_id']
        
        X_test = test_features.drop(columns=['label', 'experiment_id'])
        y_test = test_features['label']
        
        # Set current number of features for search space
        self.current_n_features = X_train.shape[1]
        
        
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Store class names in alphabetical order (as used by LabelEncoder)
        self.class_names = sorted(np.unique(y_train))
        self.n_classes = len(self.class_names)
        
        # Verify we have exactly 3 classes
        if self.n_classes != 3:
            raise ValueError(f"Expected 3 classes, but found {self.n_classes}: {self.class_names}")
        
        print(f"Detected classes (alphabetical order): {self.class_names}")
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        
                
        fold_results = []
        
        # Optimize and evaluate each model
        for model_name in self.models:
            print(f"    Optimizing {model_name.upper()}...")
            
            try:
                # Optimize hyperparameters using inner CV
                best_params, optimization_score = self.optimize_model(
                    model_name, X_train, y_train_encoded, groups_train
                )
                
                # Train final model on full training set with best parameters
                final_model = self.create_model(model_name, best_params)
                final_model.fit(X_train, y_train_encoded)
                
                # Print number of features selected
                if hasattr(final_model.named_steps['selector'], 'k'):
                    k_selected = final_model.named_steps['selector'].k
                    if k_selected == 'all':
                        n_selected = X_train.shape[1]
                    else:
                        n_selected = min(k_selected, X_train.shape[1])
                    print(f"      Features selected: {n_selected}/{X_train.shape[1]}")
                
                # Evaluate on test set
                y_pred = final_model.predict(X_test)
                y_proba = final_model.predict_proba(X_test)
                
                # Calculate comprehensive metrics
                metrics = self.calculate_comprehensive_metrics(y_test_encoded, y_pred, y_proba)
                
                auc_data = self.calculate_auc_scores(y_test_encoded, y_proba, fold_id, model_name)

                
                # Store result in exact format of your original pipeline
                result = {
                    'Model': f"{model_name.upper()}", 
                    'Fold': fold_id,
                    'Accuracy': round(metrics['accuracy'], 4),
                    'Precision_Macro': round(metrics['precision_macro'], 4),
                    'Recall_Macro': round(metrics['recall_macro'], 4),
                    'F1_Macro': round(metrics['f1_macro'], 4),
                    'Precision_Weighted': round(metrics['precision_weighted'], 4),
                    'Recall_Weighted': round(metrics['recall_weighted'], 4),
                    'F1_Weighted': round(metrics['f1_weighted'], 4),
                    'AUC_Macro': round(auc_data['macro_auc'], 4),
                    'AUC_Micro': round(auc_data['micro_auc'], 4),
                    
                    # Feature selection info
                    'Features_K': best_params.get('selector__k', 'N/A'),
                    'Score_Function': best_params.get('selector__score_func', 'N/A'),
                    
                    'Best_Params': str(best_params)  # As string for easy viewing
                }
                
                # Add per-class metrics for each class
                for class_name in self.class_names:
                    result[f'Precision_{class_name}'] = round(metrics.get(f'precision_{class_name}', 0.0), 4)
                    result[f'Recall_{class_name}'] = round(metrics.get(f'recall_{class_name}', 0.0), 4)
                    result[f'F1_{class_name}'] = round(metrics.get(f'f1_{class_name}', 0.0), 4)
                    result[f'AUC_{class_name}'] = round(auc_data.get(f'auc_{class_name}', 0.0), 4)
                    
                # Add each hyperparameter as a separate column for easier analysis
                for param_name, param_value in best_params.items():
                    result[f'HP_{param_name}'] = param_value
                
                fold_results.append(result)
                
                print(f"      Acc: {metrics['accuracy']:.4f}, F1(+): {metrics['f1_Macro']:.4f}, AUC: {metrics['auc']:.4f}")
                
            except Exception as e:
                print(f"      Error with {model_name.upper()}: {e}")
                # Store error result
                error_result = self._create_default_result(fold_id, model_name)
                fold_results.append(error_result)
        
        return fold_results

    def _create_default_result(self, fold_id: int, model_name: str) -> Dict:
        """Create default result for failed optimization."""
        
        return {
            'Model': f"{model_name.upper()}",
            'Fold': fold_id,
            'Accuracy': 0.0, 'Precision_Macro': 0.0, 'Recall_Macro': 0.0, 'F1_Macro': 0.0,
            'Precision_Weighted': 0.0, 'Recall_Weighted': 0.0, 'F1_Weighted': 0.0,
            'AUC': 0.0, 'Best_Params': 'FAILED', 'Optimization_Score': 0.0,
            'model_type': model_name.upper(), 'raw_params': {}
        }

    def evaluate_configuration(self, config_key: str, fold_results: List[Dict]) -> pd.DataFrame:
        """Evaluate a single configuration with all its folds."""
        
        print(f"\n{'='*80}")
        print(f"PROCESSING CONFIGURATION: {config_key}")
        print(f"{'='*80}")
        
        self.current_config = config_key
        self.current_config_results = []
        
        # Process each outer fold
        for fold_data in fold_results:
            fold_results_list = self.evaluate_single_fold(fold_data)
            self.current_config_results.extend(fold_results_list)
        
        # Create final results DataFrame in exact format
        final_df = self._create_exact_format_dataframe()
        
        # Store results for this configuration
        self.all_configurations_results[config_key] = final_df
        
        print(f"\n✓ Configuration {config_key} completed")
        
        return final_df

    def _create_exact_format_dataframe(self) -> pd.DataFrame:
        """Create DataFrame in exact format of your original pipeline."""
        
        # Sort results: RF folds 1-5, then XGB folds 1-5, then DT folds 1-5
        sorted_results = []
        
        for model_name in self.models:
            model_results = [r for r in self.current_config_results if r['model_name'] == model_name.upper()]
            model_results.sort(key=lambda x: x['Fold'])  # Sort by fold number
            sorted_results.extend(model_results)
        
        
        results_df = pd.DataFrame(sorted_results)
        
        # Add summary rows
        summary_data = []
        # Only include metrics that actually exist in the DataFrame
        all_metric_columns = (['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 
                            'Precision_Weighted', 'Recall_Weighted', 'F1_Weighted',
                            'AUC_Macro', 'AUC_Micro'] + 
                        [f'Precision_{class_name}' for class_name in self.class_names] +
                        [f'Recall_{class_name}' for class_name in self.class_names] +
                        [f'F1_{class_name}' for class_name in self.class_names] +
                        [f'AUC_{class_name}' for class_name in self.class_names])
        
        # Filter to only include columns that exist in the DataFrame
        metric_columns = [col for col in all_metric_columns if col in results_df.columns]
        
        # Add summary statistics (MEAN and STD for each model)
        for model_name in self.models:
            # model_data = [r for r in self.current_config_results if r['model_name'] == model_name]
            model_data = results_df[results_df['Model'] == model_name.upper()]
            
            
            if model_data.empty:
                print(f"Warning: No data found for model {model_name}")
                continue
            
            # Mean row
            mean_row = {
                'Model': model_name.upper(),
                'Fold': 'MEAN',
                'Features_K': 'N/A',
                'Score_Function': 'N/A',
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
                'Features_K': 'N/A',
                'Score_Function': 'N/A',
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

    def save_configuration_results(self, config_key: str, base_save_path: str):
        """Save results for a specific configuration."""
        
        if config_key not in self.all_configurations_results:
            print(f"Warning: No results found for configuration {config_key}")
            return
        
        config_df = self.all_configurations_results[config_key]
        
        # Save main results
        save_path = os.path.join(base_save_path, f'{config_key}_global_norm_nested_cv.csv')
        config_df.to_csv(save_path, index=False)
        
        print(f"  Main results: {save_path}")
        print(f"  Format: {len(config_df)} rows (15 individual + 6 summary)")
        
    def run_all_configurations(self, all_global_norm_results: Dict, base_save_path: str) -> Dict[str, pd.DataFrame]:
        """
        Run nested CV for all configurations and save results.
        
        Parameters:
        -----------
        all_global_norm_results : Dict
            Dictionary from your global normalization pipeline
        base_save_path : str
            Base directory for saving results
            
        Returns:
        --------
        Dict containing results for all configurations
        """
        
        print(f"\n{'='*80}")
        print("GLOBAL NORMALIZATION NESTED CV - CLASS-BASED APPROACH")
        print(f"Base save path: {base_save_path}")
        print(f"Configurations to process: {len(all_global_norm_results)}")
        print(f"{'='*80}")
        
        # Create base directory
        os.makedirs(base_save_path, exist_ok=True)
        
        # Process each configuration
        for config_key, fold_results in all_global_norm_results.items():
            # Evaluate this configuration
            config_df = self.evaluate_configuration(config_key, fold_results)
            
            # Save results
            self.save_configuration_results(config_key, base_save_path)
        
        # Create overall comparison
        self._create_overall_summary(base_save_path)
        
        print(f"\n{'='*80}")
        print("ALL CONFIGURATIONS COMPLETED")
        print(f"Results saved in exact format matching your original pipeline")
        print(f"{'='*80}")
        
        return self.all_configurations_results

    def _create_overall_summary(self, base_save_path: str):
        """Create overall comparison across all configurations."""
        
        print(f"\nCreating overall summary...")
        
        comparison_data = []
        
        for config_key, result_df in self.all_configurations_results.items():
            # Extract MEAN rows for comparison
            mean_rows = result_df[result_df['Fold'] == 'MEAN']
            
            for _, row in mean_rows.iterrows():
                comparison_data.append({
                    'Configuration': config_key,
                    'Model': row['Model'],
                    'Accuracy': row['Accuracy'],
                    'F1_Macro': row['F1_Macro'],
                    'AUC': row['AUC'],
                    'Precision_Macro': row['Precision_Macro'],
                    'Recall_Macro': row['Recall_Macro']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('F1_Macro', ascending=False)
            
            comparison_path = os.path.join(base_save_path, 'overall_configuration_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False)
            
            print(f"Overall comparison saved: {comparison_path}")
            print(f"\nTOP 5 PERFORMERS:")
            print("-" * 60)
            top_performers = comparison_df.head()[['Configuration', 'Model', 'F1_Macro', 'Accuracy', 'AUC']]
            print(top_performers.to_string(index=False))

    def get_best_configuration(self) -> Tuple[str, str, float]:
        """
        Get the best performing configuration and model.
        
        Returns:
        --------
        Tuple of (config_key, model_name, f1_macro_score)
        """
        
        best_f1 = 0.0
        best_config = None
        best_model = None
        
        for config_key, result_df in self.all_configurations_results.items():
            mean_rows = result_df[result_df['Fold'] == 'MEAN']
            
            for _, row in mean_rows.iterrows():
                if row['F1_Macro'] > best_f1:
                    best_f1 = row['F1_Macro']
                    best_config = config_key
                    best_model = row['Model']
        
        return best_config, best_model, best_f1

    def get_configuration_summary(self, config_key: str) -> pd.DataFrame:
        """Get summary statistics for a specific configuration."""
        
        if config_key not in self.all_configurations_results:
            raise ValueError(f"Configuration {config_key} not found in results")
        
        result_df = self.all_configurations_results[config_key]
        summary_rows = result_df[result_df['Fold'].isin(['MEAN', 'STD'])]
        
        return summary_rows

    def compare_configurations(self, config_keys: List[str] = None) -> pd.DataFrame:
        """
        Compare specific configurations or all configurations.
        
        Parameters:
        -----------
        config_keys : List[str], optional
            List of configuration keys to compare. If None, compares all.
            
        Returns:
        --------
        DataFrame with comparison results
        """
        
        if config_keys is None:
            config_keys = list(self.all_configurations_results.keys())
        
        comparison_data = []
        
        for config_key in config_keys:
            if config_key not in self.all_configurations_results:
                print(f"Warning: Configuration {config_key} not found")
                continue
                
            result_df = self.all_configurations_results[config_key]
            mean_rows = result_df[result_df['Fold'] == 'MEAN']
            std_rows = result_df[result_df['Fold'] == 'STD']
            
            for _, mean_row in mean_rows.iterrows():
                std_row = std_rows[std_rows['Model'] == mean_row['Model']]
                std_values = std_row.iloc[0] if not std_row.empty else None
                
                comparison_data.append({
                    'Configuration': config_key,
                    'Model': mean_row['Model'],
                    'Accuracy_Mean': mean_row['Accuracy'],
                    'Accuracy_Std': std_values['Accuracy'] if std_values is not None else 0.0,
                    'F1_Macro_Mean': mean_row['F1_Macro'],
                    'F1_Macro_Std': std_values['F1_Macro'] if std_values is not None else 0.0,
                    'AUC_Mean': mean_row['AUC'],
                    'AUC_Std': std_values['AUC'] if std_values is not None else 0.0
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('F1_Macro_Mean', ascending=False)


