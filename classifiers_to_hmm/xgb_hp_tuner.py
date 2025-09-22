# import numpy as np
# import optuna
# from sklearn.model_selection import StratifiedGroupKFold
# from xgboost import XGBClassifier
# from sklearn.metrics import log_loss
# from sklearn.preprocessing import LabelEncoder
# import pandas as pd
# import warnings
# from typing import Dict, Any, Tuple, Optional
# warnings.filterwarnings('ignore')


# class XGBoostHyperparameterTuner:
#     def __init__(self, X_train, y_train, groups_train, positive_class='void', 
#                  folds=3, n_trials=100, random_state=42):
#         """
#         XGBoost hyperparameter tuner using cross-validation.
        
#         Parameters:
#         -----------
#         X_train : pd.DataFrame or np.array
#             Training features (already preprocessed, no feature selection needed)
#         y_train : pd.Series or np.array
#             Training labels
#         groups_train : pd.Series or np.array
#             Group identifiers for GroupKFold
#         positive_class : str, optional
#             The label that should be treated as positive class for binary classification
#         folds : int
#             Number of cross-validation folds for hyperparameter optimization
#         n_trials : int
#             Number of Optuna optimization trials
#         random_state : int
#             Random state for reproducibility
#         """
#         self.X_train = X_train
#         self.y_train = y_train
#         self.groups_train = groups_train
#         self.positive_class = positive_class
#         self.folds = folds
#         self.n_trials = n_trials
#         self.random_state = random_state
        
#         self.y_encoded = self.y_train
#         self.use_label_encoder = False
#         self.positive_class_encoded = positive_class

#         # # Handle label encoding if needed
#         # if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
#         #     self.label_encoder = LabelEncoder()
#         #     self.y_encoded = self.label_encoder.fit_transform(self.y_train)
#         #     self.use_label_encoder = True
            
#         #     if positive_class:
#         #         self.positive_class_encoded = self.label_encoder.transform([positive_class])[0]
#         #         print(f"Positive class '{positive_class}' encoded as: {self.positive_class_encoded}")
#         # else:
#         #     self.y_encoded = self.y_train
#         #     self.use_label_encoder = False
#         #     self.positive_class_encoded = positive_class

#         # Cross-validation splitter
#         self.inner_cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
        
#         # Results storage
#         self.best_params = None
#         self.best_score = None
#         self.study = None

#     def get_xgb_search_space(self, trial) -> Dict[str, Any]:
#         """Define XGBoost hyperparameter search space."""
#         params = {
#             'n_estimators': trial.suggest_int('n_estimators', 50, 500),
#             'max_depth': trial.suggest_int('max_depth', 3, 12),
#             'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
#             'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#             'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
#             'gamma': trial.suggest_float('gamma', 0, 5),
#             'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
#             'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
#             'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0)
#         }
#         return params

#     def create_xgb_model(self, params: Dict[str, Any]) -> XGBClassifier:
#         """Create XGBoost model with given parameters."""
#         return XGBClassifier(
#             **params,
#             random_state=self.random_state,
#             n_jobs=-1,
#             eval_metric='logloss',
#             verbosity=0,
#             use_label_encoder=False
#         )

#     def optimize_hyperparameters(self, verbose: bool = True) -> Tuple[Dict[str, Any], float]:
#         """
#         Optimize XGBoost hyperparameters using cross-validation.
        
#         Returns:
#         --------
#         Tuple[Dict, float]: (best_parameters, best_neg_log_loss_score)
#         """
#         def objective(trial):
#             params = self.get_xgb_search_space(trial)
#             fold_scores = []

#             for train_idx, val_idx in self.inner_cv.split(self.X_train, self.y_encoded, self.groups_train):
#                 try:
#                     # Handle both DataFrame and array inputs
#                     if hasattr(self.X_train, 'iloc'):
#                         X_train_fold = self.X_train.iloc[train_idx]
#                         X_val_fold = self.X_train.iloc[val_idx]
#                     else:
#                         X_train_fold = self.X_train[train_idx]
#                         X_val_fold = self.X_train[val_idx]
                    
#                     if hasattr(self.y_encoded, 'iloc'):
#                         y_train_fold = self.y_encoded.iloc[train_idx]
#                         y_val_fold = self.y_encoded.iloc[val_idx]
#                     else:
#                         y_train_fold = self.y_encoded[train_idx]
#                         y_val_fold = self.y_encoded[val_idx]

#                     # Create and train model
#                     model = self.create_xgb_model(params)
#                     model.fit(X_train_fold, y_train_fold)

#                     # Get predicted probabilities
#                     y_proba = model.predict_proba(X_val_fold)

#                     # Calculate negative log loss (maximize this = minimize log loss)
#                     score = -log_loss(y_val_fold, y_proba)
#                     fold_scores.append(score)
                    
#                 except Exception as e:
#                     if verbose:
#                         print(f"Warning: Error in fold: {e}")
#                     # Use Optuna's pruning mechanism
#                     raise optuna.TrialPruned()

#             return np.mean(fold_scores) if fold_scores else -np.inf

#         # Create study with proper seeding for reproducibility
#         self.study = optuna.create_study(
#             direction='maximize',
#             sampler=optuna.samplers.TPESampler(
#                 seed=self.random_state,
#                 n_startup_trials=min(10, self.n_trials // 10),
#                 n_ei_candidates=24
#             ),
#             pruner=optuna.pruners.MedianPruner(
#                 n_startup_trials=min(10, self.n_trials // 10),
#                 n_warmup_steps=5,
#                 n_min_trials=1
#             )
#         )
        
#         if verbose:
#             print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
#             print(f"Using {self.folds}-fold cross-validation")
#             print(f"Training data shape: {self.X_train.shape}")
        
#         # Optimize
#         self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=verbose)
        
#         # Store results
#         self.best_params = self.study.best_params
#         self.best_score = self.study.best_value
        
#         if verbose:
#             print(f"\nOptimization completed!")
#             print(f"Best negative log loss score: {self.best_score:.6f}")
#             print(f"Best parameters: {self.best_params}")
        
#         return self.best_params, self.best_score

#     def get_best_model(self) -> XGBClassifier:
#         """
#         Get the best XGBoost model with optimized hyperparameters.
        
#         Returns:
#         --------
#         XGBClassifier: Trained model with best parameters
#         """
#         if self.best_params is None:
#             raise ValueError("Must run optimize_hyperparameters() first")
        
#         # Create model with best parameters
#         best_model = self.create_xgb_model(self.best_params)
        
#         # Train on full training data
#         best_model.fit(self.X_train, self.y_encoded)
        
#         return best_model

#     def get_optimization_summary(self) -> Dict[str, Any]:
#         """
#         Get summary of the optimization process.
        
#         Returns:
#         --------
#         Dict: Summary including best params, score, and trial information
#         """
#         if self.study is None:
#             raise ValueError("Must run optimize_hyperparameters() first")
        
#         summary = {
#             'best_params': self.best_params,
#             'best_neg_log_loss': self.best_score,
#             'n_trials_completed': len(self.study.trials),
#             'n_trials_pruned': sum(1 for trial in self.study.trials if trial.state == optuna.trial.TrialState.PRUNED),
#             'n_trials_failed': sum(1 for trial in self.study.trials if trial.state == optuna.trial.TrialState.FAIL),
#             'optimization_history': [trial.value for trial in self.study.trials if trial.value is not None]
#         }
        
#         return summary

#     def plot_optimization_history(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
#         """
#         Plot optimization history and parameter importance.
        
#         Parameters:
#         -----------
#         save_path : str, optional
#             Path to save the plot
#         figsize : tuple
#             Figure size as (width, height)
#         """
#         if self.study is None:
#             raise ValueError("Must run optimize_hyperparameters() first")
        
#         try:
#             import matplotlib.pyplot as plt
            
#             fig, axes = plt.subplots(2, 2, figsize=figsize)
            
#             # Plot 1: Optimization history
#             optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=axes[0, 0])
#             axes[0, 0].set_title('Optimization History')
            
#             # Plot 2: Parameter importance
#             optuna.visualization.matplotlib.plot_param_importances(self.study, ax=axes[0, 1])
#             axes[0, 1].set_title('Parameter Importance')
            
#             # Plot 3: Parameter relationships (parallel coordinate)
#             optuna.visualization.matplotlib.plot_parallel_coordinate(self.study, ax=axes[1, 0])
#             axes[1, 0].set_title('Parameter Relationships')
            
#             # Plot 4: Trial values distribution
#             trial_values = [trial.value for trial in self.study.trials if trial.value is not None]
#             axes[1, 1].hist(trial_values, bins=20, alpha=0.7, edgecolor='black')
#             axes[1, 1].set_title('Distribution of Trial Scores')
#             axes[1, 1].set_xlabel('Negative Log Loss')
#             axes[1, 1].set_ylabel('Frequency')
#             axes[1, 1].grid(True, alpha=0.3)
            
#             plt.tight_layout()
            
#             if save_path:
#                 plt.savefig(save_path, dpi=300, bbox_inches='tight')
#                 print(f"Optimization plots saved to: {save_path}")
#             else:
#                 plt.show()
                
#         except ImportError:
#             print("matplotlib not available for plotting")
#         except Exception as e:
#             print(f"Error creating plots: {e}")

#     def get_cv_scores_dataframe(self) -> pd.DataFrame:
#         """
#         Get detailed cross-validation scores for the best parameters.
        
#         Returns:
#         --------
#         pd.DataFrame: CV scores for each fold with best parameters
#         """
#         if self.best_params is None:
#             raise ValueError("Must run optimize_hyperparameters() first")
        
#         from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
#         cv_results = []
        
#         for fold_idx, (train_idx, val_idx) in enumerate(self.inner_cv.split(self.X_train, self.y_encoded, self.groups_train)):
#             # Handle both DataFrame and array inputs
#             if hasattr(self.X_train, 'iloc'):
#                 X_train_fold = self.X_train.iloc[train_idx]
#                 X_val_fold = self.X_train.iloc[val_idx]
#             else:
#                 X_train_fold = self.X_train[train_idx]
#                 X_val_fold = self.X_train[val_idx]
            
#             if hasattr(self.y_encoded, 'iloc'):
#                 y_train_fold = self.y_encoded.iloc[train_idx]
#                 y_val_fold = self.y_encoded.iloc[val_idx]
#             else:
#                 y_train_fold = self.y_encoded[train_idx]
#                 y_val_fold = self.y_encoded[val_idx]
            
#             # Train model with best parameters
#             model = self.create_xgb_model(self.best_params)
#             model.fit(X_train_fold, y_train_fold)
            
#             # Predictions
#             y_pred = model.predict(X_val_fold)
#             y_proba = model.predict_proba(X_val_fold)
            
#             # Calculate metrics
#             fold_result = {
#                 'fold': fold_idx + 1,
#                 'accuracy': accuracy_score(y_val_fold, y_pred),
#                 'precision_macro': precision_score(y_val_fold, y_pred, average='macro', zero_division=0),
#                 'recall_macro': recall_score(y_val_fold, y_pred, average='macro', zero_division=0),
#                 'f1_macro': f1_score(y_val_fold, y_pred, average='macro', zero_division=0),
#                 'neg_log_loss': -log_loss(y_val_fold, y_proba)
#             }
            
#             cv_results.append(fold_result)
        
#         cv_df = pd.DataFrame(cv_results)
        
#         # Add summary rows
#         summary_row = {
#             'fold': 'MEAN',
#             'accuracy': cv_df['accuracy'].mean(),
#             'precision_macro': cv_df['precision_macro'].mean(),
#             'recall_macro': cv_df['recall_macro'].mean(),
#             'f1_macro': cv_df['f1_macro'].mean(),
#             'neg_log_loss': cv_df['neg_log_loss'].mean()
#         }
        
#         std_row = {
#             'fold': 'STD',
#             'accuracy': cv_df['accuracy'].std(),
#             'precision_macro': cv_df['precision_macro'].std(),
#             'recall_macro': cv_df['recall_macro'].std(),
#             'f1_macro': cv_df['f1_macro'].std(),
#             'neg_log_loss': cv_df['neg_log_loss'].std()
#         }
        
#         cv_df = pd.concat([cv_df, pd.DataFrame([summary_row, std_row])], ignore_index=True)
        
#         return cv_df


import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import warnings
from typing import Dict, Any, Tuple, Optional
warnings.filterwarnings('ignore')


class XGBoostHyperparameterTuner:
    def __init__(self, X_train, y_train, groups_train, positive_class='void', 
                folds=3, n_trials=50, random_state=42):
        """
        XGBoost hyperparameter tuner using cross-validation.
        
        Parameters:
        -----------
        X_train : pd.DataFrame or np.array
            Training features (already preprocessed, no feature selection needed)
        y_train : pd.Series or np.array
            Training labels
        groups_train : pd.Series or np.array
            Group identifiers for GroupKFold
        positive_class : str, optional
            The label that should be treated as positive class for binary classification
        folds : int
            Number of cross-validation folds for hyperparameter optimization
        n_trials : int
            Number of Optuna optimization trials
        random_state : int
            Random state for reproducibility
        """
        self.X_train = X_train
        self.y_train = y_train
        self.groups_train = groups_train
        self.positive_class = positive_class
        self.folds = folds
        self.n_trials = n_trials
        self.random_state = random_state
        

        # # Handle label encoding if needed
        # if isinstance(y_train.iloc[0] if hasattr(y_train, 'iloc') else y_train[0], str):
        #     self.label_encoder = LabelEncoder()
        #     self.y_encoded = self.label_encoder.fit_transform(self.y_train)
        #     self.use_label_encoder = True
            
        #     if positive_class:
        #         self.positive_class_encoded = self.label_encoder.transform([positive_class])[0]
        #         print(f"Positive class '{positive_class}' encoded as: {self.positive_class_encoded}")
        # else:
        #     self.y_encoded = self.y_train
        #     self.use_label_encoder = False
        #     self.positive_class_encoded = positive_class
        
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y_train)
        
        # Identify the encoded value for the positive class
        self.positive_class_encoded = self.label_encoder.transform([self.positive_class])[0]
        print(f"Positive class '{self.positive_class}' is encoded as: {self.positive_class_encoded}")
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")
        self.use_label_encoder = False
        # Cross-validation splitter
        self.inner_cv = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=random_state)
        
        # Results storage
        self.best_params = None
        self.best_score = None
        self.study = None

    def get_xgb_search_space(self, trial) -> Dict[str, Any]:
        """Define XGBoost hyperparameter search space."""
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 2),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 2),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 0.1, 10.0)
        }
        return params

    def create_xgb_model(self, params: Dict[str, Any]) -> XGBClassifier:
        """Create XGBoost model with given parameters."""
        return XGBClassifier(
            **params,
            random_state=self.random_state,
            n_jobs=-1,
            eval_metric='logloss',
            verbosity=0,
            use_label_encoder=False
        )

    def optimize_hyperparameters(self, verbose: bool = True) -> Tuple[Dict[str, Any], float]:
        """
        Optimize XGBoost hyperparameters using cross-validation.
        
        Returns:
        --------
        Tuple[Dict, float]: (best_parameters, best_neg_log_loss_score)
        """
        def objective(trial):
            params = self.get_xgb_search_space(trial)
            fold_scores = []

            for train_idx, val_idx in self.inner_cv.split(self.X_train, self.y_encoded, self.groups_train):
                try:
                    # Handle both DataFrame and array inputs
                    if hasattr(self.X_train, 'iloc'):
                        X_train_fold = self.X_train.iloc[train_idx]
                        X_val_fold = self.X_train.iloc[val_idx]
                    else:
                        X_train_fold = self.X_train[train_idx]
                        X_val_fold = self.X_train[val_idx]
                    
                    if hasattr(self.y_encoded, 'iloc'):
                        y_train_fold = self.y_encoded.iloc[train_idx]
                        y_val_fold = self.y_encoded.iloc[val_idx]
                    else:
                        y_train_fold = self.y_encoded[train_idx]
                        y_val_fold = self.y_encoded[val_idx]

                    # Create and train model
                    model = self.create_xgb_model(params)
                    model.fit(X_train_fold, y_train_fold)

                    # Get predicted probabilities
                    y_proba = model.predict_proba(X_val_fold)

                    # Calculate negative log loss (maximize this = minimize log loss)
                    score = -log_loss(y_val_fold, y_proba)
                    fold_scores.append(score)
                    
                except Exception as e:
                    if verbose:
                        print(f"Warning: Error in fold: {e}")
                    # Use Optuna's pruning mechanism
                    raise optuna.TrialPruned()

            return np.mean(fold_scores) if fold_scores else -np.inf

        # Create study with proper seeding for reproducibility
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(
                seed=self.random_state,
                n_startup_trials=min(10, self.n_trials // 10),
                n_ei_candidates=24
            ),
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=min(10, self.n_trials // 10),
                n_warmup_steps=5,
                n_min_trials=1
            )
        )
        
        if verbose:
            print(f"Starting hyperparameter optimization with {self.n_trials} trials...")
            print(f"Using {self.folds}-fold cross-validation")
            print(f"Training data shape: {self.X_train.shape}")
        
        # Optimize
        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=verbose)
        
        # Store results
        self.best_params = self.study.best_params
        self.best_score = self.study.best_value
        
        if verbose:
            print(f"\nOptimization completed!")
            print(f"Best negative log loss score: {self.best_score:.6f}")
            print(f"Best parameters: {self.best_params}")
        
        return self.best_params, self.best_score

    def get_best_model(self) -> XGBClassifier:
        """
        Get the best XGBoost model with optimized hyperparameters.
        
        Returns:
        --------
        XGBClassifier: Trained model with best parameters
        """
        if self.best_params is None:
            raise ValueError("Must run optimize_hyperparameters() first")
        
        # Create model with best parameters
        best_model = self.create_xgb_model(self.best_params)
        
        # Train on full training data
        best_model.fit(self.X_train, self.y_encoded)
        
        return best_model

    def get_feature_selection_summary(self) -> Dict[str, Any]:
        """
        Get summary of feature selection from the optimization process.
        
        Returns:
        --------
        Dict: Summary of feature selection parameters and effectiveness
        """
        if self.best_params is None:
            raise ValueError("Must run optimize_hyperparameters() first")
        
        k_selected = self.best_params.get('selector__k', 'N/A')
        score_func = self.best_params.get('selector__score_func', 'N/A')
        
        if k_selected == 'all':
            n_features_selected = self.X_train.shape[1]
            feature_proportion = 1.0
        elif isinstance(k_selected, int):
            n_features_selected = min(k_selected, self.X_train.shape[1])
            feature_proportion = n_features_selected / self.X_train.shape[1]
        else:
            n_features_selected = 'N/A'
            feature_proportion = 'N/A'
        
        summary = {
            'k_parameter': k_selected,
            'score_function': score_func,
            'n_features_total': self.X_train.shape[1],
            'n_features_selected': n_features_selected,
            'feature_proportion': feature_proportion
        }
        
        return summary
        """
        Get summary of the optimization process.
        
        Returns:
        --------
        Dict: Summary including best params, score, and trial information
        """
        if self.study is None:
            raise ValueError("Must run optimize_hyperparameters() first")
        
        summary = {
            'best_params': self.best_params,
            'best_neg_log_loss': self.best_score,
            'n_trials_completed': len(self.study.trials),
            'n_trials_pruned': sum(1 for trial in self.study.trials if trial.state == optuna.trial.TrialState.PRUNED),
            'n_trials_failed': sum(1 for trial in self.study.trials if trial.state == optuna.trial.TrialState.FAIL),
            'optimization_history': [trial.value for trial in self.study.trials if trial.value is not None]
        }
        
        return summary

    def plot_optimization_history(self, save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot optimization history and parameter importance.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        figsize : tuple
            Figure size as (width, height)
        """
        if self.study is None:
            raise ValueError("Must run optimize_hyperparameters() first")
        
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Plot 1: Optimization history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=axes[0, 0])
            axes[0, 0].set_title('Optimization History')
            
            # Plot 2: Parameter importance
            optuna.visualization.matplotlib.plot_param_importances(self.study, ax=axes[0, 1])
            axes[0, 1].set_title('Parameter Importance')
            
            # Plot 3: Parameter relationships (parallel coordinate)
            optuna.visualization.matplotlib.plot_parallel_coordinate(self.study, ax=axes[1, 0])
            axes[1, 0].set_title('Parameter Relationships')
            
            # Plot 4: Trial values distribution
            trial_values = [trial.value for trial in self.study.trials if trial.value is not None]
            axes[1, 1].hist(trial_values, bins=20, alpha=0.7, edgecolor='black')
            axes[1, 1].set_title('Distribution of Trial Scores')
            axes[1, 1].set_xlabel('Negative Log Loss')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Optimization plots saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("matplotlib not available for plotting")
        except Exception as e:
            print(f"Error creating plots: {e}")

    def get_cv_scores_dataframe(self) -> pd.DataFrame:
        """
        Get detailed cross-validation scores for the best parameters.
        
        Returns:
        --------
        pd.DataFrame: CV scores for each fold with best parameters
        """
        if self.best_params is None:
            raise ValueError("Must run optimize_hyperparameters() first")
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        cv_results = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(self.inner_cv.split(self.X_train, self.y_encoded, self.groups_train)):
            # Handle both DataFrame and array inputs
            if hasattr(self.X_train, 'iloc'):
                X_train_fold = self.X_train.iloc[train_idx]
                X_val_fold = self.X_train.iloc[val_idx]
            else:
                X_train_fold = self.X_train[train_idx]
                X_val_fold = self.X_train[val_idx]
            
            if hasattr(self.y_encoded, 'iloc'):
                y_train_fold = self.y_encoded.iloc[train_idx]
                y_val_fold = self.y_encoded.iloc[val_idx]
            else:
                y_train_fold = self.y_encoded[train_idx]
                y_val_fold = self.y_encoded[val_idx]
            
            # Train model with best parameters
            model = self.create_xgb_model(self.best_params)
            model.fit(X_train_fold, y_train_fold)
            
            # Predictions
            y_pred = model.predict(X_val_fold)
            y_proba = model.predict_proba(X_val_fold)
            
            # Calculate metrics
            fold_result = {
                'fold': fold_idx + 1,
                'accuracy': accuracy_score(y_val_fold, y_pred),
                'precision_macro': precision_score(y_val_fold, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_val_fold, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_val_fold, y_pred, average='macro', zero_division=0),
                'neg_log_loss': -log_loss(y_val_fold, y_proba)
            }
            
            cv_results.append(fold_result)
        
        cv_df = pd.DataFrame(cv_results)
        
        # Add summary rows
        summary_row = {
            'fold': 'MEAN',
            'accuracy': cv_df['accuracy'].mean(),
            'precision_macro': cv_df['precision_macro'].mean(),
            'recall_macro': cv_df['recall_macro'].mean(),
            'f1_macro': cv_df['f1_macro'].mean(),
            'neg_log_loss': cv_df['neg_log_loss'].mean()
        }
        
        std_row = {
            'fold': 'STD',
            'accuracy': cv_df['accuracy'].std(),
            'precision_macro': cv_df['precision_macro'].std(),
            'recall_macro': cv_df['recall_macro'].std(),
            'f1_macro': cv_df['f1_macro'].std(),
            'neg_log_loss': cv_df['neg_log_loss'].std()
        }
        
        cv_df = pd.concat([cv_df, pd.DataFrame([summary_row, std_row])], ignore_index=True)
        
        return cv_df


# # Example usage function
# def tune_xgboost_example():
#     """
#     Example of how to use the XGBoostHyperparameterTuner.
#     """
#     # Example with synthetic data
#     from sklearn.datasets import make_classification
#     from sklearn.model_selection import train_test_split
    
#     # Create synthetic data
#     X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
#                              n_redundant=5, n_classes=2, random_state=42)
    
#     # Create groups (for GroupKFold)
#     groups = np.repeat(range(100), 10)  # 100 groups, 10 samples each
    
#     # Convert to DataFrame
#     X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
#     y_series = pd.Series(['class_0' if label == 0 else 'class_1' for label in y])
#     groups_series = pd.Series(groups)
    
#     # Initialize tuner
#     tuner = XGBoostHyperparameterTuner(
#         X_train=X_df, 
#         y_train=y_series, 
#         groups_train=groups_series,
#         positive_class='class_1',
#         folds=3,
#         n_trials=50,
#         random_state=42
#     )
    
#     # Optimize hyperparameters
#     best_params, best_score = tuner.optimize_hyperparameters(verbose=True)
    
#     # Get best model
#     best_model = tuner.get_best_model()
    
#     # Get CV scores
#     cv_scores = tuner.get_cv_scores_dataframe()
#     print("\nCV Scores with best parameters:")
#     print(cv_scores)
    
#     # Get optimization summary
#     summary = tuner.get_optimization_summary()
#     print(f"\nOptimization completed {summary['n_trials_completed']} trials")
#     print(f"Pruned trials: {summary['n_trials_pruned']}")
    
#     return tuner, best_model


# if __name__ == "__main__":
#     # Run example
#     tuner, model = tune_xgboost_example()