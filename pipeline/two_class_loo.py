import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
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
import seaborn as sns
warnings.filterwarnings('ignore')


class LeaveOneOutNestedCVOptimizer:
    def __init__(self, X, y, groups, positive_class="void", n_inner_folds=3,
                n_trials=50, random_state=42):
        """
        Leave-One-Out Nested Cross-Validation optimizer for binary classification.
        
        In the outer loop, leaves out one experiment (group) at a time.
        In the inner loop, uses StratifiedGroupKFold on remaining experiments.
        
        Parameters:
        -----------
        X : DataFrame
            Feature matrix
        y : Series/array
            Target labels
        groups : Series/array
            Group identifiers (experiments/subjects)
        positive_class : str
            The label that should be treated as the positive class (default: "void")
        n_inner_folds : int
            Number of folds for inner cross-validation (default: 3)
        n_trials : int
            Number of Optuna trials for hyperparameter optimization (default: 50)
        random_state : int
            Random seed for reproducibility
        """
        self.X = X
        self.y = y
        self.groups = groups
        self.positive_class = positive_class
        self.n_inner_folds = n_inner_folds
        self.n_trials = n_trials
        self.random_state = random_state

        # Get unique groups (experiments)
        self.unique_groups = sorted(self.groups.unique())
        self.n_experiments = len(self.unique_groups)
        
        print(f"Leave-One-Out CV will use {self.n_experiments} folds (one per experiment)")
        print(f"Experiments: {self.unique_groups}")

        # Global label encoder for consistency across folds
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
        
        # Identify the encoded value for the positive class
        self.positive_class_encoded = self.label_encoder.transform([self.positive_class])[0]
        print(f"Positive class '{self.positive_class}' is encoded as: {self.positive_class_encoded}")
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")

        # Cross-validation splitters
        self.outer_cv = LeaveOneGroupOut()  # Leave one experiment out
        self.inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)

        # Results storage
        self.models = ['rf', 'xgb', 'dt']
        self.outer_scores = {model: [] for model in self.models}
        self.best_params_per_fold = {model: [] for model in self.models}
        self.optimization_histories = {model: [] for model in self.models}
        self.test_experiments = []  # Track which experiment was held out
        
        # Confusion matrices and AUC storage
        self.confusion_matrices = {model: [] for model in self.models}
        self.auc_scores = {model: [] for model in self.models}
        self.roc_data = {model: [] for model in self.models}

    def get_model_search_space(self, model_name: str, trial) -> Dict[str, Any]:
        """Define model-specific hyperparameter search space with SelectKBest."""
        n_features = self.X.shape[1]
        
        # Feature selection parameters
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
        
        return params

    def create_model(self, model_name: str, params: Dict[str, Any]):
        """Create pipeline with SelectKBest and classifier."""
        clf_params = {k.replace('clf__', ''): v for k, v in params.items() if k.startswith('clf__')}
        selector_params = {k.replace('selector__', ''): v for k, v in params.items() if k.startswith('selector__')}

        if model_name == 'rf':
            clf = RandomForestClassifier(**clf_params, random_state=self.random_state, n_jobs=-1)
        elif model_name == 'xgb':
            clf = XGBClassifier(**clf_params, random_state=self.random_state, n_jobs=-1,
                                eval_metric='logloss', verbosity=0, use_label_encoder=False)
        elif model_name == 'dt':
            clf = DecisionTreeClassifier(**clf_params, random_state=self.random_state)

        score_func_map = {'f_classif': f_classif}
        score_func = score_func_map[selector_params['score_func']]
        k = selector_params['k']
        
        selector = SelectKBest(score_func=score_func, k=k)
        
        return Pipeline([
            ('selector', selector),
            ('clf', clf)
        ])

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
                'precision_positive': precision_score(y_true, y_pred, pos_label=self.positive_class_encoded, zero_division=0),
                'recall_positive': recall_score(y_true, y_pred, pos_label=self.positive_class_encoded, zero_division=0),
                'f1_positive': f1_score(y_true, y_pred, pos_label=self.positive_class_encoded, zero_division=0),
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

    def calculate_auc_scores(self, y_true, y_proba, fold_idx: int, model_name: str) -> Dict[str, Any]:
        """Calculate AUC score for binary classification."""
        auc_data = {'auc': 0.0}
        
        try:
            if y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, self.positive_class_encoded]
            else:
                y_proba_pos = y_proba.ravel()
            
            y_true_binary = (y_true == self.positive_class_encoded).astype(int)
            
            if len(np.unique(y_true_binary)) > 1:
                auc_score = roc_auc_score(y_true_binary, y_proba_pos)
                auc_data['auc'] = auc_score
                
        except Exception as e:
            print(f"Warning: Error calculating AUC for {model_name} fold {fold_idx}: {e}")
            auc_data['auc'] = 0.0
        
        return auc_data

    def calculate_roc_curves(self, y_true, y_proba, fold_idx: int, model_name: str) -> Dict[str, Any]:
        """Calculate ROC curve for visualization."""
        roc_data = {
            'fpr': np.array([0, 1]),
            'tpr': np.array([0, 0]),
            'roc_auc': 0.0
        }
        
        try:
            if y_proba.shape[1] == 2:
                y_proba_pos = y_proba[:, self.positive_class_encoded]
            else:
                y_proba_pos = y_proba.ravel()
            
            y_true_binary = (y_true == self.positive_class_encoded).astype(int)
            
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, _ = roc_curve(y_true_binary, y_proba_pos)
                roc_data['fpr'] = fpr
                roc_data['tpr'] = tpr
                roc_data['roc_auc'] = auc(fpr, tpr)
                    
        except Exception as e:
            print(f"Warning: Error calculating ROC curve for {model_name} fold {fold_idx}: {e}")
        
        return roc_data

    def calculate_confusion_matrix(self, y_true, y_pred, fold_idx: int, model_name: str) -> Dict:
        """Calculate and store confusion matrix."""
        try:
            cm = confusion_matrix(y_true, y_pred)
            
            cm_data = {
                'fold': fold_idx,
                'model': model_name,
                'confusion_matrix': cm,
                'true_labels': y_true,
                'pred_labels': y_pred,
                'test_experiment': self.test_experiments[fold_idx] if fold_idx < len(self.test_experiments) else None
            }
            
            return cm_data
            
        except Exception as e:
            print(f"Warning: Error calculating confusion matrix for {model_name} fold {fold_idx}: {e}")
            n_classes = len(np.unique(y_true)) if len(np.unique(y_true)) > 0 else 2
            empty_cm = np.zeros((n_classes, n_classes), dtype=int)
            return {
                'fold': fold_idx,
                'model': model_name,
                'confusion_matrix': empty_cm,
                'true_labels': y_true,
                'pred_labels': y_pred,
                'test_experiment': self.test_experiments[fold_idx] if fold_idx < len(self.test_experiments) else None
            }

    def optimize_model(self, model_name: str, X_train_outer, y_train_outer, groups_train_outer) -> Tuple[Dict, optuna.Study]:
        """Optimize hyperparameters using inner CV."""
        def objective(trial):
            params = self.get_model_search_space(model_name, trial)
            fold_scores = []

            # Check if we have enough groups for inner CV
            unique_train_groups = groups_train_outer.unique()
            if len(unique_train_groups) < self.n_inner_folds:
                # If not enough groups, use all data for training and validation
                print(f"Warning: Only {len(unique_train_groups)} groups available for inner CV, using simple train/validation split")
                try:
                    model = self.create_model(model_name, params)
                    model.fit(X_train_outer, y_train_outer)
                    y_proba = model.predict_proba(X_train_outer)
                    score = -log_loss(y_train_outer, y_proba)
                    return score
                except Exception as e:
                    raise optuna.TrialPruned()

            # Normal inner CV with sufficient groups
            for train_idx, val_idx in self.inner_cv.split(X_train_outer, y_train_outer, groups_train_outer):
                try:
                    X_train_inner = X_train_outer.iloc[train_idx]
                    X_val_inner = X_train_outer.iloc[val_idx]
                    y_train_inner = y_train_outer.iloc[train_idx]
                    y_val_inner = y_train_outer.iloc[val_idx]

                    model = self.create_model(model_name, params)
                    model.fit(X_train_inner, y_train_inner)
                    y_proba = model.predict_proba(X_val_inner)
                    score = -log_loss(y_val_inner, y_proba)
                    fold_scores.append(score)
                    
                except Exception as e:
                    print(f"Warning: Error in inner fold for {model_name}: {e}")
                    raise optuna.TrialPruned()

            return np.mean(fold_scores) if fold_scores else -np.inf

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

    def run_nested_cv(self) -> Dict[str, Any]:
        """Run Leave-One-Out nested cross-validation."""
        print("Starting Leave-One-Out Nested Cross-Validation...")
        print(f"Will evaluate on {self.n_experiments} experiments (one held out per fold)")

        fold_idx = 0
        for train_idx, test_idx in self.outer_cv.split(self.X, self.y_encoded, self.groups):
            # Identify which experiment is being held out
            test_experiment = self.groups.iloc[test_idx[0]]  # All test samples belong to same experiment
            self.test_experiments.append(test_experiment)
            
            print(f"\n{'='*70}")
            print(f"OUTER FOLD {fold_idx + 1}/{self.n_experiments}")
            print(f"Testing on experiment: {test_experiment}")
            print(f"{'='*70}")

            # Prepare data
            X_train_outer = self.X.iloc[train_idx]
            X_test_outer = self.X.iloc[test_idx]
            y_train_outer = pd.Series(self.y_encoded, index=self.X.index).iloc[train_idx]
            y_test_outer = pd.Series(self.y_encoded, index=self.X.index).iloc[test_idx]
            groups_train_outer = self.groups.iloc[train_idx]

            print(f"Train size: {len(X_train_outer)} (from {len(groups_train_outer.unique())} experiments)")
            print(f"Test size: {len(X_test_outer)}")
            
            # Show class distribution
            test_classes, test_counts = np.unique(y_test_outer, return_counts=True)
            test_class_names = self.label_encoder.inverse_transform(test_classes)
            print(f"Test class distribution: {dict(zip(test_class_names, test_counts))}")

            # Optimize each model
            for model_name in self.models:
                print(f"\n--- Optimizing {model_name.upper()} for experiment {test_experiment} ---")
                
                try:
                    best_params, study = self.optimize_model(model_name, X_train_outer, y_train_outer, groups_train_outer)
                    print(f"Best {model_name} params: {best_params}")
                    print(f"Best {model_name} neg_log_loss: {study.best_value:.4f}")

                    self.best_params_per_fold[model_name].append(best_params)
                    self.optimization_histories[model_name].append({
                        'best_value': study.best_value,
                        'n_trials': len(study.trials),
                        'test_experiment': test_experiment
                    })

                    # Train final model and evaluate
                    final_model = self.create_model(model_name, best_params)
                    final_model.fit(X_train_outer, y_train_outer)

                    # Show feature selection info
                    if hasattr(final_model.named_steps['selector'], 'k'):
                        k_selected = final_model.named_steps['selector'].k
                        if k_selected == 'all':
                            n_selected = X_train_outer.shape[1]
                        else:
                            n_selected = min(k_selected, X_train_outer.shape[1])
                        print(f"Features selected: {n_selected}/{X_train_outer.shape[1]}")

                    # Predictions and metrics
                    y_pred = final_model.predict(X_test_outer)
                    y_proba = final_model.predict_proba(X_test_outer)
                    
                    # Store results
                    cm_data = self.calculate_confusion_matrix(y_test_outer, y_pred, fold_idx, model_name)
                    self.confusion_matrices[model_name].append(cm_data)
                    
                    fold_metrics = self.calculate_metrics(y_test_outer, y_pred)
                    fold_metrics['test_experiment'] = test_experiment
                    self.outer_scores[model_name].append(fold_metrics)
                    
                    auc_data = self.calculate_auc_scores(y_test_outer, y_proba, fold_idx, model_name)
                    auc_data['test_experiment'] = test_experiment
                    self.auc_scores[model_name].append(auc_data)
                    
                    roc_data = self.calculate_roc_curves(y_test_outer, y_proba, fold_idx, model_name)
                    roc_data['test_experiment'] = test_experiment
                    self.roc_data[model_name].append(roc_data)

                    print(f"{model_name} - Accuracy: {fold_metrics['accuracy']:.4f}, "
                          f"F1(pos): {fold_metrics['f1_positive']:.4f}, "
                          f"AUC: {auc_data['auc']:.4f}")

                except Exception as e:
                    print(f"Error optimizing {model_name}: {e}")
                    # Store defaults for failed optimization
                    self.best_params_per_fold[model_name].append({})
                    self.optimization_histories[model_name].append({
                        'best_value': 0.0, 'n_trials': 0, 'test_experiment': test_experiment
                    })
                    
                    default_metrics = {
                        'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                        'precision_weighted': 0.0, 'recall_weighted': 0.0, 'f1_weighted': 0.0,
                        'precision_positive': 0.0, 'recall_positive': 0.0, 'f1_positive': 0.0,
                        'precision_negative': 0.0, 'recall_negative': 0.0, 'f1_negative': 0.0,
                        'test_experiment': test_experiment
                    }
                    self.outer_scores[model_name].append(default_metrics)
                    
                    self.auc_scores[model_name].append({'auc': 0.0, 'test_experiment': test_experiment})
                    
                    default_roc = {
                        'fpr': np.array([0, 1]), 'tpr': np.array([0, 0]), 'roc_auc': 0.0,
                        'test_experiment': test_experiment
                    }
                    self.roc_data[model_name].append(default_roc)

            fold_idx += 1

        return self._summarize_results()

    def _summarize_results(self) -> Dict[str, Any]:
        """Summarize Leave-One-Out results."""
        print(f"\n{'='*80}")
        print("LEAVE-ONE-OUT NESTED CROSS-VALIDATION RESULTS")
        print(f"Evaluated across {self.n_experiments} experiments")
        print(f"Positive class: '{self.positive_class}'")
        print(f"{'='*80}")

        summaries = {}
        
        for model_name in self.models:
            if self.outer_scores[model_name]:
                scores_df = pd.DataFrame(self.outer_scores[model_name])
                auc_df = pd.DataFrame(self.auc_scores[model_name])
                
                summaries[model_name] = {
                    'mean_scores': scores_df.select_dtypes(include=[np.number]).mean().to_dict(),
                    'std_scores': scores_df.select_dtypes(include=[np.number]).std().to_dict(),
                    'mean_auc_scores': auc_df.select_dtypes(include=[np.number]).mean().to_dict(),
                    'std_auc_scores': auc_df.select_dtypes(include=[np.number]).std().to_dict(),
                    'all_fold_scores': self.outer_scores[model_name],
                    'all_auc_scores': self.auc_scores[model_name],
                    'test_experiments': self.test_experiments,
                    'best_params_per_fold': self.best_params_per_fold[model_name],
                    'optimization_history': self.optimization_histories[model_name]
                }
                
                print(f"\n{model_name.upper()} Results:")
                print("-" * 50)
                
                key_metrics = ['accuracy', 'f1_positive', 'precision_positive', 'recall_positive', 'f1_macro']
                for metric in key_metrics:
                    if metric in scores_df.columns:
                        mean_val = scores_df[metric].mean()
                        std_val = scores_df[metric].std()
                        print(f"{metric:18}: {mean_val:.4f} ± {std_val:.4f}")
                
                print(f"{'auc':18}: {auc_df['auc'].mean():.4f} ± {auc_df['auc'].std():.4f}")
                
                # Show per-experiment results
                print(f"\nPer-experiment results:")
                for i, exp in enumerate(self.test_experiments):
                    if i < len(scores_df):
                        acc = scores_df.iloc[i]['accuracy']
                        f1 = scores_df.iloc[i]['f1_positive']
                        auc_score = auc_df.iloc[i]['auc']
                        print(f"  {exp}: Acc={acc:.3f}, F1={f1:.3f}, AUC={auc_score:.3f}")

        # Find best model
        if summaries:
            best_model = max(summaries.keys(), 
                        key=lambda x: summaries[x]['mean_scores'].get('f1_positive', 0))
            best_f1 = summaries[best_model]['mean_scores']['f1_positive']
            best_acc = summaries[best_model]['mean_scores']['accuracy']
            best_auc = summaries[best_model]['mean_auc_scores']['auc']
            
            print(f"\n{'='*80}")
            print(f"BEST MODEL: {best_model.upper()}")
            print(f"F1 Score (positive): {best_f1:.4f}")
            print(f"Accuracy: {best_acc:.4f}")
            print(f"AUC: {best_auc:.4f}")
            print(f"{'='*80}")
            
            summaries['best_model'] = best_model
            summaries['best_f1_positive'] = best_f1
            summaries['best_accuracy'] = best_acc
            summaries['best_auc'] = best_auc

        return summaries

    def get_results_dataframe(self) -> pd.DataFrame:
        """Create comprehensive results DataFrame with per-experiment details."""
        results_data = []
        
        for model_name in self.models:
            for fold_idx in range(len(self.test_experiments)):
                fold_metrics = self.outer_scores[model_name][fold_idx] if fold_idx < len(self.outer_scores[model_name]) else {}
                fold_auc = self.auc_scores[model_name][fold_idx] if fold_idx < len(self.auc_scores[model_name]) else {}
                best_params = self.best_params_per_fold[model_name][fold_idx] if fold_idx < len(self.best_params_per_fold[model_name]) else {}
                
                row = {
                    'Model': model_name.upper(),
                    'Fold': fold_idx + 1,
                    'Test_Experiment': self.test_experiments[fold_idx],
                    'Accuracy': round(fold_metrics.get('accuracy', 0.0), 4),
                    'F1_Positive': round(fold_metrics.get('f1_positive', 0.0), 4),
                    'Precision_Positive': round(fold_metrics.get('precision_positive', 0.0), 4),
                    'Recall_Positive': round(fold_metrics.get('recall_positive', 0.0), 4),
                    'F1_Macro': round(fold_metrics.get('f1_macro', 0.0), 4),
                    'AUC': round(fold_auc.get('auc', 0.0), 4),
                    'Features_K': best_params.get('selector__k', 'N/A'),
                    'Best_Params': str(best_params)
                }
                
                results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        
        # Add summary statistics
        summary_data = []
        metric_columns = ['Accuracy', 'F1_Positive', 'Precision_Positive', 'Recall_Positive', 'F1_Macro', 'AUC']
        
        for model_name in self.models:
            model_data = results_df[results_df['Model'] == model_name.upper()]
            if not model_data.empty:
                # Mean row
                mean_row = {
                    'Model': model_name.upper(),
                    'Fold': 'MEAN',
                    'Test_Experiment': 'ALL',
                    'Features_K': 'N/A',
                    'Best_Params': 'N/A'
                }
                for metric in metric_columns:
                    if metric in model_data.columns:
                        mean_row[metric] = round(model_data[metric].mean(), 4)
                summary_data.append(mean_row)
                
                # Std row
                std_row = {
                    'Model': model_name.upper(),
                    'Fold': 'STD',
                    'Test_Experiment': 'ALL',
                    'Features_K': 'N/A',
                    'Best_Params': 'N/A'
                }
                for metric in metric_columns:
                    if metric in model_data.columns:
                        std_row[metric] = round(model_data[metric].std(), 4)
                summary_data.append(std_row)
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            final_df = pd.concat([results_df, summary_df], ignore_index=True)
        else:
            final_df = results_df
            
        return final_df

    def get_experiment_performance_summary(self) -> pd.DataFrame:
        """Create a summary showing how each experiment performed as test set."""
        summary_data = []
        
        for exp_idx, experiment in enumerate(self.test_experiments):
            row = {'Experiment': experiment}
            
            for model_name in self.models:
                if exp_idx < len(self.outer_scores[model_name]):
                    metrics = self.outer_scores[model_name][exp_idx]
                    auc_data = self.auc_scores[model_name][exp_idx] if exp_idx < len(self.auc_scores[model_name]) else {}
                    
                    row[f'{model_name.upper()}_Accuracy'] = round(metrics.get('accuracy', 0.0), 4)
                    row[f'{model_name.upper()}_F1_Positive'] = round(metrics.get('f1_positive', 0.0), 4)
                    row[f'{model_name.upper()}_AUC'] = round(auc_data.get('auc', 0.0), 4)
            
            summary_data.append(row)
        
        return pd.DataFrame(summary_data)

    def get_feature_selection_summary(self) -> pd.DataFrame:
        """Summary of feature selection across all folds."""
        summary_data = []
        
        for model_name in self.models:
            for fold_idx, params in enumerate(self.best_params_per_fold[model_name]):
                if params:
                    k_selected = params.get('selector__k', 'N/A')
                    score_func = params.get('selector__score_func', 'N/A')
                    
                    if k_selected == 'all':
                        n_features_selected = self.X.shape[1]
                    elif isinstance(k_selected, int):
                        n_features_selected = min(k_selected, self.X.shape[1])
                    else:
                        n_features_selected = 'N/A'
                    
                    summary_data.append({
                        'Model': model_name.upper(),
                        'Fold': fold_idx + 1,
                        'Test_Experiment': self.test_experiments[fold_idx] if fold_idx < len(self.test_experiments) else 'N/A',
                        'K_Parameter': k_selected,
                        'N_Features_Selected': n_features_selected,
                        'Score_Function': score_func,
                        'Feature_Proportion': round(n_features_selected / self.X.shape[1], 3) if isinstance(n_features_selected, int) else 'N/A'
                    })
        
        return pd.DataFrame(summary_data)

    def get_confusion_matrix_dataframe(self) -> pd.DataFrame:
        """Detailed confusion matrix data for all models and experiments."""
        cm_data = []
        class_labels = [f'non_{self.positive_class}', self.positive_class]
        
        for model_name in self.models:
            for cm_info in self.confusion_matrices[model_name]:
                fold_idx = cm_info['fold']
                cm = cm_info['confusion_matrix']
                test_exp = cm_info.get('test_experiment', 'Unknown')
                
                row = {
                    'Model': model_name.upper(),
                    'Fold': fold_idx + 1,
                    'Test_Experiment': test_exp
                }
                
                # Add confusion matrix cells
                for i in range(2):
                    for j in range(2):
                        if i < cm.shape[0] and j < cm.shape[1]:
                            row[f'CM_{class_labels[i]}_pred_{class_labels[j]}'] = cm[i, j]
                        else:
                            row[f'CM_{class_labels[i]}_pred_{class_labels[j]}'] = 0
                
                # Add derived binary classification metrics
                if cm.size == 4:
                    tn, fp, fn, tp = cm.ravel()
                    row.update({
                        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
                        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0
                    })
                
                cm_data.append(row)
        
        return pd.DataFrame(cm_data)

    def plot_experiment_performance_heatmap(self, save_path: str = None, figsize: Tuple[int, int] = (12, 8)):
        """Plot heatmap showing performance of each model on each experiment."""
        exp_summary = self.get_experiment_performance_summary()
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        metrics = ['Accuracy', 'F1_Positive', 'AUC']
        
        for idx, metric in enumerate(metrics):
            # Prepare data matrix
            data_matrix = []
            for model in self.models:
                col_name = f'{model.upper()}_{metric}'
                if col_name in exp_summary.columns:
                    data_matrix.append(exp_summary[col_name].values)
            
            if data_matrix:
                data_matrix = np.array(data_matrix)
                
                # Create heatmap
                im = axes[idx].imshow(data_matrix, cmap='RdYlBu_r', aspect='auto')
                
                # Set labels
                axes[idx].set_xticks(range(len(self.test_experiments)))
                axes[idx].set_xticklabels(self.test_experiments, rotation=45, ha='right')
                axes[idx].set_yticks(range(len(self.models)))
                axes[idx].set_yticklabels([m.upper() for m in self.models])
                axes[idx].set_title(f'{metric} by Experiment')
                
                # Add colorbar
                cbar = plt.colorbar(im, ax=axes[idx])
                cbar.set_label(metric)
                
                # Add text annotations
                for i in range(len(self.models)):
                    for j in range(len(self.test_experiments)):
                        if i < data_matrix.shape[0] and j < data_matrix.shape[1]:
                            text = axes[idx].text(j, i, f'{data_matrix[i, j]:.3f}',
                                            ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Experiment performance heatmap saved to: {save_path}")
        else:
            plt.show()

    def plot_roc_curves(self, save_path: str = None, figsize: Tuple[int, int] = (15, 5)):
        """Plot ROC curves for all models, showing individual experiments."""
        fig, axes = plt.subplots(1, len(self.models), figsize=figsize)
        if len(self.models) == 1:
            axes = [axes]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.test_experiments)))
        
        for model_idx, model in enumerate(self.models):
            ax = axes[model_idx]
            
            fold_aucs = []
            for fold_idx, (fold_data, exp) in enumerate(zip(self.roc_data[model], self.test_experiments)):
                if fold_data['roc_auc'] > 0:
                    ax.plot(fold_data['fpr'], fold_data['tpr'], 
                           color=colors[fold_idx], alpha=0.7, linewidth=1.5,
                           label=f'{exp} (AUC={fold_data["roc_auc"]:.3f})')
                    fold_aucs.append(fold_data['roc_auc'])
            
            # Calculate and plot mean ROC
            if fold_aucs and len(fold_aucs) > 1:
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
                           label=f'Mean (AUC={mean_auc:.3f}±{std_auc:.3f})')
            
            # Diagonal line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model.upper()}: ROC Curves (Leave-One-Out)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        else:
            plt.show()

    def plot_performance_by_experiment(self, save_path: str = None, figsize: Tuple[int, int] = (12, 8)):
        """Plot performance metrics for each experiment."""
        exp_summary = self.get_experiment_performance_summary()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        metrics = ['Accuracy', 'F1_Positive', 'AUC']
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            x_pos = np.arange(len(self.test_experiments))
            width = 0.25
            
            for model_idx, model in enumerate(self.models):
                col_name = f'{model.upper()}_{metric}'
                if col_name in exp_summary.columns:
                    values = exp_summary[col_name].values
                    ax.bar(x_pos + model_idx * width, values, width, 
                          label=model.upper(), alpha=0.8)
            
            ax.set_xlabel('Test Experiment')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} by Test Experiment')
            ax.set_xticks(x_pos + width)
            ax.set_xticklabels(self.test_experiments, rotation=45, ha='right')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Fourth subplot: Summary statistics
        ax = axes[3]
        summary_stats = []
        for model in self.models:
            model_data = []
            for metric in metrics:
                col_name = f'{model.upper()}_{metric}'
                if col_name in exp_summary.columns:
                    mean_val = exp_summary[col_name].mean()
                    model_data.append(mean_val)
            summary_stats.append(model_data)
        
        if summary_stats:
            x_pos = np.arange(len(metrics))
            for model_idx, model in enumerate(self.models):
                ax.plot(x_pos, summary_stats[model_idx], 'o-', 
                       label=model.upper(), linewidth=2, markersize=8)
            
            ax.set_xlabel('Metric')
            ax.set_ylabel('Mean Value')
            ax.set_title('Mean Performance Across All Experiments')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance by experiment plots saved to: {save_path}")
        else:
            plt.show()