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
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Any, Tuple
from sklearn.metrics import confusion_matrix
import os
import seaborn as sns
warnings.filterwarnings('ignore')

class ModifiedNestedCVOptimizer:
    """
    Nested cross-validation optimizer adapted for global normalization with pre-split data.
    
    This class adapts your original NestedCVOptimizer to work with pre-split data from 
    global normalization while maintaining the exact same output format.
    """
    
    def __init__(self, positive_class="void", n_inner_folds=3, n_trials=50, random_state=42):
        """
        Initialize the optimizer for pre-split global normalization data.
        
        Parameters:
        -----------
        positive_class : str
            The label that should be treated as the positive class (default: "void")
        n_inner_folds : int
            Number of folds for inner cross-validation (hyperparameter optimization)
        n_trials : int
            Number of optimization trials for each model
        random_state : int
            Random state for reproducibility
        """
        self.positive_class = positive_class
        self.n_inner_folds = n_inner_folds
        self.n_trials = n_trials
        self.random_state = random_state
        
        # Inner CV splitter
        self.inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
        
        # Models to evaluate
        self.models = ['rf', 'xgb', 'dt']
        
        self.confusion_matrices = {model: [] for model in self.models}        
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
                                eval_metric='logloss', verbosity=0, use_label_encoder=False)
        elif model_name == 'dt':
            clf = DecisionTreeClassifier(**clf_params, random_state=self.random_state)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Create the SelectKBest instance
        score_func_map = {'f_classif': f_classif}
        score_func = score_func_map[selector_params['score_func']]
        k = selector_params['k']
        
        selector = SelectKBest(score_func=score_func, k=k)

        # Create and return the pipeline
        pipeline = Pipeline([
            ('selector', selector),
            ('clf', clf)
        ])
        return pipeline

    def calculate_comprehensive_metrics(self, y_true, y_pred, y_proba, positive_class_encoded) -> Dict[str, float]:
        """Calculate comprehensive binary classification metrics exactly like your original."""
        
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
                'precision_positive': precision_score(y_true, y_pred, pos_label=positive_class_encoded, zero_division=0),
                'recall_positive': recall_score(y_true, y_pred, pos_label=positive_class_encoded, zero_division=0),
                'f1_positive': f1_score(y_true, y_pred, pos_label=positive_class_encoded, zero_division=0),
                
                # Binary classification metrics for negative class (non-void)
                'precision_negative': precision_score(y_true, y_pred, pos_label=1-positive_class_encoded, zero_division=0),
                'recall_negative': recall_score(y_true, y_pred, pos_label=1-positive_class_encoded, zero_division=0),
                'f1_negative': f1_score(y_true, y_pred, pos_label=1-positive_class_encoded, zero_division=0)
            }
            
            # Calculate AUC
            if len(np.unique(y_true)) > 1 and y_proba is not None:
                if y_proba.shape[1] == 2:
                    y_proba_pos = y_proba[:, positive_class_encoded]
                else:
                    y_proba_pos = y_proba.ravel()
                
                y_true_binary = (y_true == positive_class_encoded).astype(int)
                auc_score = roc_auc_score(y_true_binary, y_proba_pos)
                metrics['auc'] = auc_score
            else:
                metrics['auc'] = 0.0
            
            return metrics
            
        except Exception as e:
            print(f"Warning: Error calculating metrics: {e}")
            return {
                'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                'precision_weighted': 0.0, 'recall_weighted': 0.0, 'f1_weighted': 0.0,
                'precision_positive': 0.0, 'recall_positive': 0.0, 'f1_positive': 0.0,
                'precision_negative': 0.0, 'recall_negative': 0.0, 'f1_negative': 0.0,
                'auc': 0.0
            }

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

    def calculate_confusion_matrix(self, y_true, y_pred, fold_idx: int, model_name: str) -> np.ndarray:
        """
        Calculate and store confusion matrix for the current fold.
        
        Args:
            y_true: True labels (encoded)
            y_pred: Predicted labels (encoded)
            fold_idx: Current fold index
            model_name: Name of the model
            
        Returns:
            Confusion matrix as numpy array
        """
        try:
            # Calculate confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            
            # Store with metadata
            cm_data = {
                'fold': fold_idx,
                'model': model_name,
                'confusion_matrix': cm,
                'true_labels': y_true,
                'pred_labels': y_pred
            }
            
            return cm_data
            
        except Exception as e:
            print(f"Warning: Error calculating confusion matrix for {model_name} fold {fold_idx}: {e}")
            # Return empty matrix with appropriate shape
            n_classes = len(np.unique(y_true)) if len(np.unique(y_true)) > 0 else 2
            empty_cm = np.zeros((n_classes, n_classes), dtype=int)
            return {
                'fold': fold_idx,
                'model': model_name,
                'confusion_matrix': empty_cm,
                'true_labels': y_true,
                'pred_labels': y_pred
            }
            
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
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Identify the encoded value for the positive class
        positive_class_encoded = label_encoder.transform([self.positive_class])[0]
        
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
                
                # Calculate and store confusion matrix
                cm_data = self.calculate_confusion_matrix(y_test_encoded, y_pred, fold_id, model_name)
                self.confusion_matrices[model_name].append(cm_data)
                
                # Calculate comprehensive metrics
                metrics = self.calculate_comprehensive_metrics(y_test_encoded, y_pred, y_proba, positive_class_encoded)
                
                # Store result in exact format of your original pipeline
                result = {
                    'Model': f"{model_name.upper()}",  # e.g., "1RF", "2XGB"
                    'Fold': fold_id,
                    'Accuracy': round(metrics['accuracy'], 4),
                    'Precision_Macro': round(metrics['precision_macro'], 4),
                    'Recall_Macro': round(metrics['recall_macro'], 4),
                    'F1_Macro': round(metrics['f1_macro'], 4),
                    'Precision_Weighted': round(metrics['precision_weighted'], 4),
                    'Recall_Weighted': round(metrics['recall_weighted'], 4),
                    'F1_Weighted': round(metrics['f1_weighted'], 4),
                    'Precision_Positive': round(metrics['precision_positive'], 4),
                    'Recall_Positive': round(metrics['recall_positive'], 4),
                    'F1_Positive': round(metrics['f1_positive'], 4),
                    'Precision_Negative': round(metrics['precision_negative'], 4),
                    'Recall_Negative': round(metrics['recall_negative'], 4),
                    'F1_Negative': round(metrics['f1_negative'], 4),
                    
                    
                    # Feature selection info
                    'Features_K': best_params.get('selector__k', 'N/A'),
                    'Score_Function': best_params.get('selector__score_func', 'N/A'),
                    
                    'Best_Params': str(best_params)  # As string for easy viewing
                }
                
                # Add each hyperparameter as a separate column for easier analysis
                for param_name, param_value in best_params.items():
                    result[f'HP_{param_name}'] = param_value
                
                
                fold_results.append(result)
                
                print(f"      Acc: {metrics['accuracy']:.4f}, F1(+): {metrics['f1_positive']:.4f}")
                
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
            'Precision_Positive': 0.0, 'Recall_Positive': 0.0, 'F1_Positive': 0.0,
            'Precision_Negative': 0.0, 'Recall_Negative': 0.0, 'F1_Negative': 0.0,
            'Best_Params': 'FAILED', 'Optimization_Score': 0.0,
            'model_name': model_name.upper(), 'raw_params': {}
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
            model_results = [r for r in self.current_config_results if r['Model'] == model_name.upper()]
            model_results.sort(key=lambda x: x['Fold'])  # Sort by fold number
            sorted_results.extend(model_results)
            
        results_df = pd.DataFrame(sorted_results)
        
        
        # Add summary rows
        summary_data = []
        # Only include metrics that actually exist in the DataFrame
        all_metric_columns = (['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 
                            'Precision_Weighted', 'Recall_Weighted', 'F1_Weighted','Precision_Positive',
                            'Recall_Positive', 'F1_Positive', 'Precision_Negative', 'Recall_Negative',
                            'F1_Negative'                         
                            ] 
                        )
        
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
                    'F1_Positive': row['F1_Positive'],
                    'Precision_Positive': row['Precision_Positive'],
                    'Recall_Positive': row['Recall_Positive']
                })
        
        if comparison_data:
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.sort_values('F1_Positive', ascending=False)
            
            comparison_path = os.path.join(base_save_path, 'overall_configuration_comparison.csv')
            comparison_df.to_csv(comparison_path, index=False)
            
            print(f"Overall comparison saved: {comparison_path}")
            print(f"\nTOP 5 PERFORMERS:")
            print("-" * 60)
            top_performers = comparison_df.head()[['Configuration', 'Model', 'F1_Positive', 'Accuracy']]
            print(top_performers.to_string(index=False))

    def get_best_configuration(self) -> Tuple[str, str, float]:
        """
        Get the best performing configuration and model.
        
        Returns:
        --------
        Tuple of (config_key, model_name, f1_positive_score)
        """
        
        best_f1 = 0.0
        best_config = None
        best_model = None
        
        for config_key, result_df in self.all_configurations_results.items():
            mean_rows = result_df[result_df['Fold'] == 'MEAN']
            
            for _, row in mean_rows.iterrows():
                if row['F1_Positive'] > best_f1:
                    best_f1 = row['F1_Positive']
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
                    'F1_Positive_Mean': mean_row['F1_Positive'],
                    'F1_Positive_Std': std_values['F1_Positive'] if std_values is not None else 0.0,
                    'AUC_Mean': mean_row['AUC'],
                    'AUC_Std': std_values['AUC'] if std_values is not None else 0.0
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.sort_values('F1_Positive_Mean', ascending=False)

    def plot_confusion_matrices(self, save_path: str = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot confusion matrices for all models.
        Shows both individual fold matrices and aggregated matrices.
        """
        n_models = len(self.models)
        
        # Create subplots: 2 rows (individual folds + aggregated), n_models columns
        fig, axes = plt.subplots(2, n_models, figsize=figsize)
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        # For 3-class implementation, use class names
        if hasattr(self, 'class_names'):
            class_labels = self.class_names

        for model_idx, model_name in enumerate(self.models):
            if not self.confusion_matrices[model_name]:
                continue
                
            # Plot 1: Individual fold matrices (small multiples)
            cms = [cm_data['confusion_matrix'] for cm_data in self.confusion_matrices[model_name]]
            
            if cms:
                # Show aggregated confusion matrix (sum across folds)
                aggregated_cm = np.sum(cms, axis=0)
                
                # Normalize for display (showing percentages)
                cm_normalized = aggregated_cm.astype('float') / aggregated_cm.sum(axis=1)[:, np.newaxis]
                
                # Plot aggregated confusion matrix
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                        xticklabels=class_labels, yticklabels=class_labels,
                        ax=axes[0, model_idx], cbar_kws={'label': 'Proportion'})
                axes[0, model_idx].set_title(f'{model_name.upper()}: Aggregated CM\n(Normalized)')
                axes[0, model_idx].set_xlabel('Predicted')
                axes[0, model_idx].set_ylabel('True')
                
                # Plot aggregated confusion matrix with raw counts
                sns.heatmap(aggregated_cm, annot=True, fmt='d', cmap='Oranges',
                        xticklabels=class_labels, yticklabels=class_labels,
                        ax=axes[1, model_idx], cbar_kws={'label': 'Count'})
                axes[1, model_idx].set_title(f'{model_name.upper()}: Aggregated CM\n(Raw Counts)')
                axes[1, model_idx].set_xlabel('Predicted')
                axes[1, model_idx].set_ylabel('True')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to: {save_path}")
        else:
            plt.show()
            
    
    def plot_individual_fold_cms(self, model_name: str, save_path: str = None, figsize: Tuple[int, int] = (12, 4)):
        """
        Plot confusion matrices for each fold of a specific model.
        """
        if model_name not in self.confusion_matrices or not self.confusion_matrices[model_name]:
            print(f"No confusion matrices available for model: {model_name}")
            return
        
        n_folds = len(self.confusion_matrices[model_name])
        fig, axes = plt.subplots(1, n_folds, figsize=figsize)
        if n_folds == 1:
            axes = [axes]
        
        # Determine class labels
        if hasattr(self, 'class_names'):
            class_labels = self.class_names
        else:
            class_labels = [f'Not_{self.positive_class}', self.positive_class]
            
        
        for fold_idx, cm_data in enumerate(self.confusion_matrices[model_name]):
            cm = cm_data['confusion_matrix']
            
            # Normalize confusion matrix
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_labels, yticklabels=class_labels,
                    ax=axes[fold_idx], cbar=fold_idx == n_folds-1)
            axes[fold_idx].set_title(f'Fold {fold_idx + 1}')
            axes[fold_idx].set_xlabel('Predicted')
            if fold_idx == 0:
                axes[fold_idx].set_ylabel('True')
        
        plt.suptitle(f'{model_name.upper()}: Confusion Matrices by Fold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Individual fold confusion matrices saved to: {save_path}")
        else:
            plt.show()
