import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
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
warnings.filterwarnings('ignore')
import seaborn as sns


class ModifiedNestedCVOptimizer:
    """
    Modified NestedCVOptimizer for pre-split data from global normalization
    
    This version accepts pre-split train/test data for each fold and only performs
    inner CV for hyperparameter optimization on the training data.
    """
    
    def __init__(self, positive_class="void", n_inner_folds=3, n_trials=50, random_state=42):
        """
        Initialize the modified optimizer for pre-split data.
        
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

        # Cross-validation splitter for inner CV only
        self.inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)

        # Results storage
        self.models = ['rf', 'xgb', 'dt']
        self.all_results = []  # Store results from all folds
        
    def get_model_search_space(self, model_name: str, trial) -> Dict[str, Any]:
        """Define model-specific hyperparameter search space with SelectKBest."""
        # Choose number of features to select (k parameter)
        n_features = self.current_n_features  # Set during optimization
        
        feature_selection_k = trial.suggest_categorical('selector__k', 
                                                       [int(n_features * 0.1), 
                                                        int(n_features * 0.2),
                                                        int(n_features * 0.3),
                                                        int(n_features * 0.4),
                                                        int(n_features * 0.5),
                                                        int(n_features * 0.7),
                                                        int(n_features * 0.8),
                                                        int(n_features * 0.9),
                                                        'all'])
        
        # Choose scoring function for feature selection
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

    def calculate_metrics(self, y_true, y_pred, y_proba, positive_class_encoded) -> Dict[str, float]:
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
                'precision_positive': precision_score(y_true, y_pred, pos_label=positive_class_encoded, zero_division=0),
                'recall_positive': recall_score(y_true, y_pred, pos_label=positive_class_encoded, zero_division=0),
                'f1_positive': f1_score(y_true, y_pred, pos_label=positive_class_encoded, zero_division=0),
                
                # Binary classification metrics for negative class (non-void) for completeness
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
            default_metrics = {
                'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                'precision_weighted': 0.0, 'recall_weighted': 0.0, 'f1_weighted': 0.0,
                'precision_positive': 0.0, 'recall_positive': 0.0, 'f1_positive': 0.0,
                'precision_negative': 0.0, 'recall_negative': 0.0, 'f1_negative': 0.0,
                'auc': 0.0
            }
            return default_metrics
            
    def optimize_model(self, model_name: str, X_train, y_train, groups_train) -> Tuple[Dict, optuna.Study]:
        """Optimize hyperparameters for a specific model using inner CV."""
        
        # Set current number of features for search space
        self.current_n_features = X_train.shape[1]
        
        def objective(trial):
            params = self.get_model_search_space(model_name, trial)
            fold_scores = []

            for train_idx, val_idx in self.inner_cv.split(X_train, y_train, groups_train):
                try:
                    X_train_inner = X_train.iloc[train_idx]
                    X_val_inner = X_train.iloc[val_idx]
                    y_train_inner = y_train.iloc[train_idx]
                    y_val_inner = y_train.iloc[val_idx]

                    model = self.create_model(model_name, params)
                    model.fit(X_train_inner, y_train_inner)

                    # Get predicted probabilities for the validation set
                    y_proba = model.predict_proba(X_val_inner)

                    # Calculate negative log loss (maximize this = minimize log loss)
                    score = -log_loss(y_val_inner, y_proba)
                    fold_scores.append(score)
                    
                except Exception as e:
                    print(f"Warning: Error in inner fold for {model_name}: {e}")
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
        
        return study.best_params, study
    
    def evaluate_single_fold(self, fold_data: Dict, config_key: str) -> List[Dict]:
        """
        Evaluate all models on a single fold's pre-split data.
        
        Parameters:
        -----------
        fold_data : Dict
            Dictionary containing 'train_features', 'test_features', 'fold', etc.
        config_key : str
            Configuration identifier (e.g., "2s_0.5")
            
        Returns:
        --------
        List of result dictionaries, one for each model
        """
        fold_id = fold_data['fold']
        train_features = fold_data['train_features']
        test_features = fold_data['test_features']
        
        print(f"\n{'='*50}")
        print(f"FOLD {fold_id} - CONFIG: {config_key}")
        print(f"{'='*50}")
        
        # Prepare data
        X_train = train_features.drop(columns=['label', 'experiment_id'])
        y_train = train_features['label']
        groups_train = train_features['experiment_id']
        
        X_test = test_features.drop(columns=['label', 'experiment_id'])
        y_test = test_features['label']
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Identify the encoded value for the positive class
        positive_class_encoded = label_encoder.transform([self.positive_class])[0]
        
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"Train labels: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"Test labels: {pd.Series(y_test).value_counts().to_dict()}")
        print(f"Positive class '{self.positive_class}' encoded as: {positive_class_encoded}")
        
        fold_results = []
        
        # Optimize and evaluate each model
        for model_name in self.models:
            print(f"\n--- Optimizing {model_name.upper()} ---")
            
            try:
                # Optimize hyperparameters using inner CV on training data
                best_params, study = self.optimize_model(
                    model_name, X_train, pd.Series(y_train_encoded), groups_train
                )
                
                print(f"Best {model_name} params: {best_params}")
                print(f"Best {model_name} neg_log_loss: {study.best_value:.4f}")
                
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
                    print(f"Features selected: {n_selected}/{X_train.shape[1]}")
                
                # Evaluate on test set
                y_pred = final_model.predict(X_test)
                y_proba = final_model.predict_proba(X_test)
                
                # Calculate comprehensive metrics
                metrics = self.calculate_metrics(y_test_encoded, y_pred, y_proba, positive_class_encoded)
                
                # Store results
                result = {
                    'config': config_key,
                    'fold': fold_id,
                    'model': model_name.upper(),
                    'best_params': best_params,
                    'optimization_score': study.best_value,
                    'n_trials': len(study.trials),
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    **metrics  # Unpack all metrics
                }
                
                fold_results.append(result)
                
                print(f"{model_name} Results:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1 (positive): {metrics['f1_positive']:.4f}")
                print(f"  AUC: {metrics['auc']:.4f}")
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                # Store error result
                error_result = {
                    'config': config_key,
                    'fold': fold_id,
                    'model': model_name.upper(),
                    'best_params': {},
                    'optimization_score': 0.0,
                    'n_trials': 0,
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'accuracy': 0.0, 'f1_positive': 0.0, 'auc': 0.0,
                    'precision_positive': 0.0, 'recall_positive': 0.0,
                    'error': str(e)
                }
                fold_results.append(error_result)
        
        return fold_results
    
    def evaluate_all_folds(self, all_global_norm_results: Dict) -> pd.DataFrame:
        """
        Evaluate all models on all folds and configurations.
        
        Parameters:
        -----------
        all_global_norm_results : Dict
            Dictionary from your global normalization pipeline
            
        Returns:
        --------
        DataFrame with all results
        """
        print("Starting evaluation of all folds with hyperparameter optimization...")
        
        for config_key, fold_results in all_global_norm_results.items():
            print(f"\n{'='*80}")
            print(f"PROCESSING CONFIGURATION: {config_key}")
            print(f"{'='*80}")
            
            for fold_data in fold_results:
                fold_results_list = self.evaluate_single_fold(fold_data, config_key)
                self.all_results.extend(fold_results_list)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.all_results)
        
        print(f"\n{'='*80}")
        print("EVALUATION COMPLETED")
        print(f"{'='*80}")
        print(f"Total evaluations: {len(results_df)}")
        print(f"Configurations: {len(results_df['config'].unique())}")
        print(f"Models: {results_df['model'].unique().tolist()}")
        
        return results_df
    
    def get_summary_statistics(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Generate summary statistics across folds for each configuration and model."""
        
        # Group by config and model, calculate mean and std
        summary_stats = []
        
        for config in results_df['config'].unique():
            for model in results_df['model'].unique():
                config_model_data = results_df[
                    (results_df['config'] == config) & (results_df['model'] == model)
                ]
                
                if len(config_model_data) > 0:
                    metrics = ['accuracy', 'f1_positive', 'auc', 'precision_positive', 'recall_positive']
                    
                    summary = {
                        'config': config,
                        'model': model,
                        'n_folds': len(config_model_data)
                    }
                    
                    for metric in metrics:
                        if metric in config_model_data.columns:
                            summary[f'{metric}_mean'] = config_model_data[metric].mean()
                            summary[f'{metric}_std'] = config_model_data[metric].std()
                    
                    summary_stats.append(summary)
        
        return pd.DataFrame(summary_stats)


def run_modified_nested_cv(all_global_norm_results, positive_class="void", n_inner_folds=3, n_trials=50):
    """
    Convenience function to run the modified nested CV on your global normalization results.
    
    Parameters:
    -----------
    all_global_norm_results : Dict
        Results from your global normalization pipeline
    positive_class : str
        Positive class label
    n_inner_folds : int
        Number of inner CV folds for hyperparameter optimization
    n_trials : int
        Number of optimization trials per model
        
    Returns:
    --------
    Tuple of (detailed_results_df, summary_df, optimizer_instance)
    """
    
    # Initialize the modified optimizer
    optimizer = ModifiedNestedCVOptimizer(
        positive_class=positive_class,
        n_inner_folds=n_inner_folds,
        n_trials=n_trials,
        random_state=42
    )
    
    # Run evaluation on all folds
    detailed_results = optimizer.evaluate_all_folds(all_global_norm_results)
    
    # Generate summary statistics
    summary_results = optimizer.get_summary_statistics(detailed_results)
    
    return detailed_results, summary_results, optimizer