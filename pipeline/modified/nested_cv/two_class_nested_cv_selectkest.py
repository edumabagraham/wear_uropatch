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
warnings.filterwarnings('ignore')


class NestedCVOptimizer:
    def __init__(self, X, y, groups, positive_class="void", n_outer_folds=3, n_inner_folds=2,
                n_trials=50, random_state=42):
        """
        Nested cross-validation optimizer for binary classification
        with RandomForest, XGBoost, and DecisionTree including AUC analysis.
        
        Uses SelectKBest for feature selection instead of PCA.
        Each model gets its own hyperparameter optimization in each outer fold.
        
        Parameters:
        -----------
        positive_class : str
            The label that should be treated as the positive class (default: "void")
        """
        self.X = X
        self.y = y
        self.groups = groups
        self.positive_class = positive_class
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.n_trials = n_trials
        self.random_state = random_state

        # Global label encoder for consistency across folds
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Identify the encoded value for the positive class
        self.positive_class_encoded = self.label_encoder.transform([self.positive_class])[0]
        print(f"Positive class '{self.positive_class}' is encoded as: {self.positive_class_encoded}")
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")

        # Cross-validation splitters
        self.outer_cv = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
        self.inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)

        # Results storage
        self.models = ['rf', 'xgb', 'dt']
        self.outer_scores = {model: [] for model in self.models}
        self.best_params_per_fold = {model: [] for model in self.models}
        self.optimization_histories = {model: [] for model in self.models}
        
        # AUC storage
        self.auc_scores = {model: [] for model in self.models}  # Store AUC scores for each fold
        self.roc_data = {model: [] for model in self.models}  # Store ROC curves for plotting

    def get_model_search_space(self, model_name: str, trial) -> Dict[str, Any]:
        """Define model-specific hyperparameter search space with SelectKBest."""
        # SelectKBest parameters
        n_features = self.X.shape[1]
        
        # Choose number of features to select (k parameter)
        # Can be integer or 'all' to use all features
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
        score_func = trial.suggest_categorical('selector__score_func', 
                                            ['f_classif'])
        
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
        Calculate AUC score for binary classification.
        
        Args:
            y_true: True labels (encoded)
            y_proba: Predicted probabilities for each class
            fold_idx: Current fold index
            model_name: Name of the model
            
        Returns:
            Dictionary containing AUC score
        """
        auc_data = {
            'auc': 0.0
        }
        
        try:
            # For binary classification, get probabilities for positive class
            if y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, self.positive_class_encoded]
            else:
                # Handle case where predict_proba returns single column
                y_proba_pos = y_proba.ravel()
            
            # Create binary labels (1 for positive class, 0 for negative)
            y_true_binary = (y_true == self.positive_class_encoded).astype(int)
            
            # Calculate AUC
            if len(np.unique(y_true_binary)) > 1:  # Need both classes present
                auc_score = roc_auc_score(y_true_binary, y_proba_pos)
                auc_data['auc'] = auc_score
            else:
                auc_data['auc'] = 0.0
                
        except Exception as e:
            print(f"Warning: Error calculating AUC for {model_name} fold {fold_idx}: {e}")
            auc_data['auc'] = 0.0
        
        return auc_data

    def calculate_roc_curves(self, y_true, y_proba, fold_idx: int, model_name: str) -> Dict[str, Any]:
        """
        Calculate ROC curve for binary classification visualization.
        """
        roc_data = {
            'fpr': np.array([0, 1]),
            'tpr': np.array([0, 0]),
            'roc_auc': 0.0
        }
        
        try:
            # For binary classification, get probabilities for positive class
            if y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, self.positive_class_encoded]
            else:
                y_proba_pos = y_proba.ravel()
            
            # Create binary labels
            y_true_binary = (y_true == self.positive_class_encoded).astype(int)
            
            # Calculate ROC curve
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba_pos)
                roc_data['fpr'] = fpr
                roc_data['tpr'] = tpr
                roc_data['roc_auc'] = auc(fpr, tpr)
                    
        except Exception as e:
            print(f"Warning: Error calculating ROC curve for {model_name} fold {fold_idx}: {e}")
        
        return roc_data

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
            
    def optimize_model(self, model_name: str, X_train_outer, y_train_outer, groups_train_outer) -> Tuple[Dict, optuna.Study]:
        """Optimize hyperparameters for a specific model using inner CV with neg_log_loss."""
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

                    # Fit the model (the pipeline) on the inner training set
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
                seed=self.random_state,  # Critical for reproducibility
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

    def run_nested_cv(self) -> Dict[str, Any]:
        """Run the complete nested CV optimization for all models."""
        print("Starting Binary Nested Cross-Validation with SelectKBest feature selection...")
        print(f"Positive class: '{self.positive_class}' (encoded as {self.positive_class_encoded})")
        print(f"Total features available: {self.X.shape[1]}")

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
                    print(f"Best {model_name} neg_log_loss score: {study.best_value:.4f}")

                    # Store optimization results
                    self.best_params_per_fold[model_name].append(best_params)
                    self.optimization_histories[model_name].append({
                        'best_value': study.best_value,
                        'n_trials': len(study.trials)
                    })

                    # Train final model on full outer training set
                    final_model = self.create_model(model_name, best_params)
                    final_model.fit(X_train_outer, y_train_outer)

                    # Print number of features selected
                    if hasattr(final_model.named_steps['selector'], 'k'):
                        k_selected = final_model.named_steps['selector'].k
                        if k_selected == 'all':
                            n_selected = X_train_outer.shape[1]
                        else:
                            n_selected = min(k_selected, X_train_outer.shape[1])
                        print(f"Features selected: {n_selected}/{X_train_outer.shape[1]}")

                    # Evaluate on outer test set
                    y_pred = final_model.predict(X_test_outer)
                    y_proba = final_model.predict_proba(X_test_outer)
                    
                    # Calculate standard metrics
                    fold_metrics = self.calculate_metrics(y_test_outer, y_pred)
                    self.outer_scores[model_name].append(fold_metrics)
                    
                    # Calculate AUC scores
                    auc_data = self.calculate_auc_scores(y_test_outer, y_proba, fold_idx, model_name)
                    self.auc_scores[model_name].append(auc_data)
                    
                    # Store ROC data for potential plotting
                    roc_data = self.calculate_roc_curves(y_test_outer, y_proba, fold_idx, model_name)
                    self.roc_data[model_name].append(roc_data)

                    print(f"{model_name} test accuracy: {fold_metrics['accuracy']:.4f}")
                    print(f"{model_name} test F1 (positive): {fold_metrics['f1_positive']:.4f}")
                    print(f"{model_name} test AUC: {auc_data['auc']:.4f}")

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
                    
                    # Store default AUC data
                    self.auc_scores[model_name].append({'auc': 0.0})
                    
                    # Store default ROC data
                    default_roc = {
                        'fpr': np.array([0, 1]),
                        'tpr': np.array([0, 0]),
                        'roc_auc': 0.0
                    }
                    self.roc_data[model_name].append(default_roc)

        return self._summarize_results()

    def _summarize_results(self) -> Dict[str, Any]:
        """Summarize and display results."""
        print(f"\n{'='*80}")
        print("BINARY NESTED CROSS-VALIDATION RESULTS SUMMARY")
        print(f"Positive class: '{self.positive_class}'")
        print(f"Feature selection method: SelectKBest")
        print(f"{'='*80}")

        summaries = {}
        
        for model_name in self.models:
            if self.outer_scores[model_name]:
                scores_df = pd.DataFrame(self.outer_scores[model_name])
                auc_df = pd.DataFrame(self.auc_scores[model_name])
                
                summaries[model_name] = {
                    'mean_scores': scores_df.mean(numeric_only=True).to_dict(),
                    'std_scores': scores_df.std(numeric_only=True).to_dict(),
                    'mean_auc_scores': auc_df.mean(numeric_only=True).to_dict(),
                    'std_auc_scores': auc_df.std(numeric_only=True).to_dict(),
                    'all_fold_scores': self.outer_scores[model_name],
                    'all_auc_scores': self.auc_scores[model_name],
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
                
                # Print AUC
                print(f"{'auc':18}: {auc_df['auc'].mean():.4f} ± {auc_df['auc'].std():.4f}")
                
                print(f"Individual fold accuracies: {[f'{score:.4f}' for score in scores_df['accuracy'].tolist()]}")
                print(f"Individual fold F1 (pos): {[f'{score:.4f}' for score in scores_df['f1_positive'].tolist()]}")
                print(f"Individual fold AUCs: {[f'{score:.4f}' for score in auc_df['auc'].tolist()]}")

        # Find best model based on F1 score of positive class
        if summaries:
            best_model = max(summaries.keys(), 
                        key=lambda x: summaries[x]['mean_scores'].get('f1_positive', 0))
            best_f1_positive = summaries[best_model]['mean_scores']['f1_positive']
            best_accuracy = summaries[best_model]['mean_scores']['accuracy']
            best_auc = summaries[best_model]['mean_auc_scores']['auc']
            
            print(f"\n{'='*80}")
            print(f"BEST MODEL: {best_model.upper()}")
            print(f"F1 Score (positive class): {best_f1_positive:.4f}")
            print(f"Accuracy: {best_accuracy:.4f}")
            print(f"AUC: {best_auc:.4f}")
            print(f"{'='*80}")
            
            summaries['best_model'] = best_model
            summaries['best_f1_positive'] = best_f1_positive
            summaries['best_accuracy'] = best_accuracy
            summaries['best_auc'] = best_auc

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
    
    def get_fold_auc_results(self) -> Dict[str, List[float]]:
        """
        Get the AUC scores for each model across all outer folds.
        """
        fold_aucs = {}
        for model_name in self.models:
            aucs = [fold_scores['auc'] for fold_scores in self.auc_scores[model_name]]
            fold_aucs[model_name] = aucs
        
        return fold_aucs
    
    def get_feature_selection_summary(self) -> pd.DataFrame:
        """
        Create a summary of feature selection choices across folds and models.
        """
        summary_data = []
        
        for model_name in self.models:
            for fold_idx, params in enumerate(self.best_params_per_fold[model_name]):
                if params:  # Check if params is not empty
                    k_selected = params.get('selector__k', 'N/A')
                    score_func = params.get('selector__score_func', 'N/A')
                    
                    # Calculate actual number of features if k is 'all'
                    if k_selected == 'all':
                        n_features_selected = self.X.shape[1]
                    elif isinstance(k_selected, int):
                        n_features_selected = min(k_selected, self.X.shape[1])
                    else:
                        n_features_selected = 'N/A'
                    
                    summary_data.append({
                        'Model': model_name.upper(),
                        'Fold': fold_idx + 1,
                        'K_Parameter': k_selected,
                        'N_Features_Selected': n_features_selected,
                        'Score_Function': score_func,
                        'Feature_Proportion': round(n_features_selected / self.X.shape[1], 3) if isinstance(n_features_selected, int) else 'N/A'
                    })
        
        return pd.DataFrame(summary_data)
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with fold results including binary classification metrics
        and AUC scores for the positive class ('void').
        """
        results_data = []
        
        # Add fold results
        for model_name in self.models:
            for fold_idx in range(self.n_outer_folds):
                # Get all metrics for this fold
                fold_metrics = self.outer_scores[model_name][fold_idx] if fold_idx < len(self.outer_scores[model_name]) else {}
                
                # Get AUC scores for this fold
                fold_auc = self.auc_scores[model_name][fold_idx] if fold_idx < len(self.auc_scores[model_name]) else {}
                
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
                    
                    # AUC score
                    'AUC': round(fold_auc.get('auc', 0.0), 4),
                    
                    # Feature selection info
                    'Features_K': best_params.get('selector__k', 'N/A'),
                    'Score_Function': best_params.get('selector__score_func', 'N/A'),
                    
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
        # Only include metrics that actually exist in the DataFrame
        all_metric_columns = ['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 
                            'Precision_Weighted', 'Recall_Weighted', 'F1_Weighted',
                            'Precision_Positive', 'Recall_Positive', 'F1_Positive',
                            'Precision_Negative', 'Recall_Negative', 'F1_Negative',
                            'AUC']
        
        # Filter to only include columns that exist in the DataFrame
        metric_columns = [col for col in all_metric_columns if col in results_df.columns]
        
        for model_name in self.models:
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

    def get_auc_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame focused on AUC scores across all models.
        """
        summary_data = []
        
        for model_name in self.models:
            if self.auc_scores[model_name]:
                auc_df = pd.DataFrame(self.auc_scores[model_name])
                
                summary_data.append({
                    'Model': model_name.upper(),
                    'AUC_Mean': round(auc_df['auc'].mean(), 4),
                    'AUC_Std': round(auc_df['auc'].std(), 4),
                    'AUC_Min': round(auc_df['auc'].min(), 4),
                    'AUC_Max': round(auc_df['auc'].max(), 4),
                    'Individual_Folds': [round(x, 4) for x in auc_df['auc'].tolist()]
                })
        
        return pd.DataFrame(summary_data)

    def plot_roc_curves(self, save_path: str = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot ROC curves for all models across folds.
        
        Args:
            save_path: Path to save the plot. If None, displays the plot.
            figsize: Figure size as (width, height)
        """
        fig, axes = plt.subplots(1, len(self.models), figsize=figsize)
        if len(self.models) == 1:
            axes = [axes]
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for model_idx, model in enumerate(self.models):
            if not self.roc_data[model]:
                continue
                
            ax = axes[model_idx]
            
            # Plot ROC curve for each fold
            fold_aucs = []
            for fold_idx, fold_data in enumerate(self.roc_data[model]):
                if fold_data['roc_auc'] > 0:
                    ax.plot(fold_data['fpr'], fold_data['tpr'], 
                        color=colors[fold_idx % len(colors)], 
                        alpha=0.6, linewidth=1,
                        label=f'Fold {fold_idx + 1} (AUC = {fold_data["roc_auc"]:.3f})')
                    fold_aucs.append(fold_data['roc_auc'])
            
            # Calculate and plot mean ROC curve
            if fold_aucs:
                # Interpolate all curves to common FPR points
                mean_fpr = np.linspace(0, 1, 100)
                tprs = []
                
                for fold_data in self.roc_data[model]:
                    if fold_data['roc_auc'] > 0:
                        interp_tpr = np.interp(mean_fpr, fold_data['fpr'], fold_data['tpr'])
                        tprs.append(interp_tpr)
                
                if tprs:
                    mean_tpr = np.mean(tprs, axis=0)
                    mean_auc = np.mean(fold_aucs)
                    std_auc = np.std(fold_aucs)
                    
                    ax.plot(mean_fpr, mean_tpr, 'k-', linewidth=3,
                        label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
                    
                    # Add confidence interval
                    std_tpr = np.std(tprs, axis=0)
                    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, 
                                color='grey', alpha=0.2)
            
            # Plot diagonal line
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model.upper()}: ROC Curves\n({self.positive_class} Detection)')
            ax.legend(loc="lower right", fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        else:
            plt.show()

    def plot_auc_comparison(self, save_path: str = None, figsize: Tuple[int, int] = (10, 6)):
        """
        Plot AUC scores comparison across models and folds.
        """
        # Prepare data for plotting
        plot_data = []
        for model_name in self.models:
            if self.auc_scores[model_name]:
                for fold_idx, fold_auc in enumerate(self.auc_scores[model_name]):
                    plot_data.append({
                        'Model': model_name.upper(),
                        'AUC': fold_auc['auc'],
                        'Fold': fold_idx + 1
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot 1: AUC across folds
        for model in self.models:
            model_data = plot_df[plot_df['Model'] == model.upper()]
            if not model_data.empty:
                axes[0].plot(model_data['Fold'], model_data['AUC'], 'o-', 
                        label=model.upper(), linewidth=2, markersize=6)
        axes[0].set_title('AUC Across Folds (SelectKBest)')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('AUC')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # Plot 2: Mean AUC comparison (bar plot)
        mean_aucs = []
        std_aucs = []
        model_names = []
        for model in self.models:
            if self.auc_scores[model]:
                auc_df = pd.DataFrame(self.auc_scores[model])
                mean_aucs.append(auc_df['auc'].mean())
                std_aucs.append(auc_df['auc'].std())
                model_names.append(model.upper())
        
        if mean_aucs:
            bars = axes[1].bar(model_names, mean_aucs, yerr=std_aucs, 
                            capsize=5, alpha=0.7)
            axes[1].set_xlabel('Model')
            axes[1].set_ylabel('Mean AUC')
            axes[1].set_title('Mean AUC Comparison (SelectKBest)')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim([0, 1])
            
            # Add value labels on bars
            for bar, mean_auc, std_auc in zip(bars, mean_aucs, std_aucs):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + std_auc + 0.01,
                        f'{mean_auc:.3f}±{std_auc:.3f}',
                        ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"AUC comparison plots saved to: {save_path}")
        else:
            plt.show()

    def plot_feature_selection_analysis(self, save_path: str = None, figsize: Tuple[int, int] = (14, 8)):
        """
        Plot analysis of feature selection choices across models and folds.
        """
        feature_summary = self.get_feature_selection_summary()
        
        if feature_summary.empty:
            print("No feature selection data available for plotting.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Number of features selected by model and fold
        for model in self.models:
            model_data = feature_summary[feature_summary['Model'] == model.upper()]
            if not model_data.empty and 'N_Features_Selected' in model_data.columns:
                # Filter out non-numeric values
                numeric_data = model_data[pd.to_numeric(model_data['N_Features_Selected'], errors='coerce').notna()]
                if not numeric_data.empty:
                    axes[0, 0].plot(numeric_data['Fold'], 
                                pd.to_numeric(numeric_data['N_Features_Selected']), 
                                'o-', label=model.upper(), linewidth=2, markersize=6)
        
        axes[0, 0].set_title('Number of Features Selected Across Folds')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Feature proportion by model
        mean_proportions = []
        model_names = []
        for model in self.models:
            model_data = feature_summary[feature_summary['Model'] == model.upper()]
            if not model_data.empty and 'Feature_Proportion' in model_data.columns:
                numeric_proportions = pd.to_numeric(model_data['Feature_Proportion'], errors='coerce')
                if not numeric_proportions.isna().all():
                    mean_proportions.append(numeric_proportions.mean())
                    model_names.append(model.upper())
        
        if mean_proportions:
            axes[0, 1].bar(model_names, mean_proportions, alpha=0.7)
            axes[0, 1].set_title('Mean Feature Proportion Selected')
            axes[0, 1].set_xlabel('Model')
            axes[0, 1].set_ylabel('Proportion of Features')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylim([0, 1])
        
        # Plot 3: Score function usage
        if 'Score_Function' in feature_summary.columns:
            score_func_counts = feature_summary['Score_Function'].value_counts()
            axes[1, 0].pie(score_func_counts.values, labels=score_func_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Score Function Usage Distribution')
        
        # Plot 4: Feature selection vs performance correlation
        # Combine with AUC scores if available
        if self.auc_scores:
            correlation_data = []
            for model in self.models:
                model_features = feature_summary[feature_summary['Model'] == model.upper()]
                model_aucs = [fold['auc'] for fold in self.auc_scores[model]] if self.auc_scores[model] else []
                
                for idx, (_, row) in enumerate(model_features.iterrows()):
                    if idx < len(model_aucs) and pd.to_numeric(row['Feature_Proportion'], errors='coerce') is not np.nan:
                        correlation_data.append({
                            'Feature_Proportion': pd.to_numeric(row['Feature_Proportion'], errors='coerce'),
                            'AUC': model_aucs[idx],
                            'Model': row['Model']
                        })
            
            if correlation_data:
                corr_df = pd.DataFrame(correlation_data)
                for model in self.models:
                    model_corr = corr_df[corr_df['Model'] == model.upper()]
                    if not model_corr.empty:
                        axes[1, 1].scatter(model_corr['Feature_Proportion'], model_corr['AUC'], 
                                        label=model.upper(), alpha=0.7, s=50)
                
                axes[1, 1].set_title('Feature Proportion vs AUC Performance')
                axes[1, 1].set_xlabel('Proportion of Features Selected')
                axes[1, 1].set_ylabel('AUC Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature selection analysis saved to: {save_path}")
        else:
            plt.show()