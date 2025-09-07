import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_curve, auc, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, label_binarize
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Any, Tuple
warnings.filterwarnings('ignore')


class NestedCVOptimizer:
    def __init__(self, X, y, groups, n_outer_folds=5, n_inner_folds=3,
                 n_trials=50, random_state=42):
        """
        Improved nested cross-validation optimizer for multi-class classification
        with RandomForest, XGBoost, and DecisionTree including AUC analysis.
        
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
        
        # Store class names in alphabetical order (as used by LabelEncoder)
        self.class_names = sorted(np.unique(self.y))
        self.n_classes = len(self.class_names)
        print(f"Detected classes (alphabetical order): {self.class_names}")

        # Cross-validation splitters
        self.outer_cv = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
        self.inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)

        # Results storage - now properly structured per model per fold
        self.models = ['rf', 'xgb', 'dt']
        self.outer_scores = {model: [] for model in self.models}
        self.best_params_per_fold = {model: [] for model in self.models}
        self.optimization_histories = {model: [] for model in self.models}
        
        # AUC storage
        self.auc_scores = {model: [] for model in self.models}  # Store AUC scores for each fold
        self.roc_data = {model: [] for model in self.models}  # Store ROC curves for plotting

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

    def calculate_roc_curves(self, y_true, y_proba, fold_idx: int, model_name: str) -> Dict[str, Any]:
        """
        Calculate ROC curves for visualization (optional - for plotting later).
        """
        roc_data = {
            'fold': fold_idx,
            'model': model_name,
            'fpr': {},
            'tpr': {},
            'roc_auc': {}
        }
        
        try:
            # Binarize the output for multi-class ROC
            y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
            
            # Handle case where test set doesn't contain all classes
            if y_true_bin.shape[1] < self.n_classes:
                missing_classes = self.n_classes - y_true_bin.shape[1]
                padding = np.zeros((y_true_bin.shape[0], missing_classes))
                y_true_bin = np.hstack([y_true_bin, padding])
            
            # Calculate ROC curve for each class
            for i, class_name in enumerate(self.class_names):
                if i < y_proba.shape[1] and np.sum(y_true_bin[:, i]) > 0:
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                    roc_data['fpr'][class_name] = fpr
                    roc_data['tpr'][class_name] = tpr
                    roc_data['roc_auc'][class_name] = auc(fpr, tpr)
                else:
                    # Handle missing class
                    roc_data['fpr'][class_name] = np.array([0, 1])
                    roc_data['tpr'][class_name] = np.array([0, 0])
                    roc_data['roc_auc'][class_name] = 0.0
                    
        except Exception as e:
            print(f"Warning: Error calculating ROC curves for {model_name} fold {fold_idx}: {e}")
            for class_name in self.class_names:
                roc_data['fpr'][class_name] = np.array([0, 1])
                roc_data['tpr'][class_name] = np.array([0, 0])
                roc_data['roc_auc'][class_name] = 0.0
        
        return roc_data

    def calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate comprehensive classification metrics including per-class metrics with proper class names."""
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
            default_metrics = {'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0}
            for class_name in self.class_names:
                default_metrics[f'precision_{class_name}'] = 0.0
                default_metrics[f'recall_{class_name}'] = 0.0
                default_metrics[f'f1_{class_name}'] = 0.0
            return default_metrics

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
                    print(f"{model_name} macro AUC: {auc_data['macro_auc']:.4f}")
                    print(f"{model_name} micro AUC: {auc_data['micro_auc']:.4f}")
                    
                    # Print per-class metrics for this fold
                    for class_name in self.class_names:
                        if f'recall_{class_name}' in fold_metrics:
                            recall = fold_metrics[f'recall_{class_name}']
                            auc_score = auc_data.get(f'auc_{class_name}', 0.0)
                            print(f"{model_name} {class_name} - Recall: {recall:.4f}, AUC: {auc_score:.4f}")

                except Exception as e:
                    print(f"Error optimizing {model_name}: {e}")
                    # Store default results for failed optimization
                    self.best_params_per_fold[model_name].append({})
                    self.optimization_histories[model_name].append({'best_value': 0.0, 'n_trials': 0})
                    default_metrics = self.calculate_metrics([0], [0])  # This will return proper defaults
                    self.outer_scores[model_name].append(default_metrics)
                    
                    # Store default AUC data
                    default_auc = {'macro_auc': 0.0, 'micro_auc': 0.0}
                    for class_name in self.class_names:
                        default_auc[f'auc_{class_name}'] = 0.0
                    self.auc_scores[model_name].append(default_auc)
                    
                    # Store default ROC data
                    default_roc = {
                        'fold': fold_idx,
                        'model': model_name,
                        'fpr': {class_name: np.array([0, 1]) for class_name in self.class_names},
                        'tpr': {class_name: np.array([0, 0]) for class_name in self.class_names},
                        'roc_auc': {class_name: 0.0 for class_name in self.class_names}
                    }
                    self.roc_data[model_name].append(default_roc)

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
                auc_df = pd.DataFrame(self.auc_scores[model_name])
                
                summaries[model_name] = {
                    'mean_scores': scores_df.mean().to_dict(),
                    'std_scores': scores_df.std().to_dict(),
                    # 'mean_auc_scores': auc_df.mean().to_dict(),
                    # 'std_auc_scores': auc_df.std().to_dict(),
                    'mean_auc_scores': auc_df.mean(numeric_only=True).to_dict(),
                    'std_auc_scores': auc_df.std(numeric_only=True).to_dict(),
                    'all_fold_scores': self.outer_scores[model_name],
                    'all_auc_scores': self.auc_scores[model_name],
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
                        print(f"{metric:15}: {mean_val:.4f} ± {std_val:.4f}")
                
                # Print AUC scores
                print(f"{'macro_auc':15}: {auc_df['macro_auc'].mean():.4f} ± {auc_df['macro_auc'].std():.4f}")
                print(f"{'micro_auc':15}: {auc_df['micro_auc'].mean():.4f} ± {auc_df['micro_auc'].std():.4f}")
                
                # Print per-class recall and AUC summary
                print("\nPer-class Performance:")
                for class_name in self.class_names:
                    recall_col = f'recall_{class_name}'
                    auc_col = f'auc_{class_name}'
                    if recall_col in scores_df.columns and auc_col in auc_df.columns:
                        recall_mean = scores_df[recall_col].mean()
                        recall_std = scores_df[recall_col].std()
                        auc_mean = auc_df[auc_col].mean()
                        auc_std = auc_df[auc_col].std()
                        print(f"{class_name:12} - Recall: {recall_mean:.4f}±{recall_std:.4f}, AUC: {auc_mean:.4f}±{auc_std:.4f}")
                
                print(f"Individual fold accuracies: {[f'{score:.4f}' for score in scores_df['accuracy'].tolist()]}")
                print(f"Individual fold macro AUCs: {[f'{score:.4f}' for score in auc_df['macro_auc'].tolist()]}")

        # Find best model (you can choose between accuracy and AUC)
        if summaries:
            best_model_acc = max(summaries.keys(), 
                               key=lambda x: summaries[x]['mean_scores'].get('accuracy', 0))
            best_model_auc = max(summaries.keys(), 
                               key=lambda x: summaries[x]['mean_auc_scores'].get('macro_auc', 0))
            
            best_accuracy = summaries[best_model_acc]['mean_scores']['accuracy']
            best_auc = summaries[best_model_auc]['mean_auc_scores']['macro_auc']
            
            print(f"\n{'='*80}")
            print(f"BEST MODEL (Accuracy): {best_model_acc.upper()} ({best_accuracy:.4f})")
            print(f"BEST MODEL (Macro AUC): {best_model_auc.upper()} ({best_auc:.4f})")
            print(f"{'='*80}")
            
            summaries['best_model_accuracy'] = best_model_acc
            summaries['best_model_auc'] = best_model_auc
            summaries['best_accuracy'] = best_accuracy
            summaries['best_macro_auc'] = best_auc

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
    
    def get_fold_auc_results(self) -> Dict[str, List[float]]:
        """
        Get the macro AUC scores for each model across all outer folds.
        """
        fold_aucs = {}
        for model_name in self.models:
            aucs = [fold_scores['macro_auc'] for fold_scores in self.auc_scores[model_name]]
            fold_aucs[model_name] = aucs
        
        return fold_aucs
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with fold accuracies, macro metrics, per-class metrics, 
        AUC scores, and best hyperparameters. Returns a clean DataFrame ready for analysis.
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
                    'AUC_Macro': round(fold_auc.get('macro_auc', 0.0), 4),
                    'AUC_Micro': round(fold_auc.get('micro_auc', 0.0), 4),
                    'Best_Params': str(best_params)  # As string for easy viewing
                }
                
                # Add per-class metrics for each class
                for class_name in self.class_names:
                    row[f'Precision_{class_name}'] = round(fold_metrics.get(f'precision_{class_name}', 0.0), 4)
                    row[f'Recall_{class_name}'] = round(fold_metrics.get(f'recall_{class_name}', 0.0), 4)
                    row[f'F1_{class_name}'] = round(fold_metrics.get(f'f1_{class_name}', 0.0), 4)
                    row[f'AUC_{class_name}'] = round(fold_auc.get(f'auc_{class_name}', 0.0), 4)
                
                # Add each hyperparameter as a separate column for easier analysis
                for param_name, param_value in best_params.items():
                    row[f'HP_{param_name}'] = param_value
                
                results_data.append(row)
        
        # Create DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Add summary rows
        summary_data = []
        metric_columns = (['Accuracy', 'Precision_Macro', 'Recall_Macro', 'F1_Macro', 
                          'Precision_Weighted', 'Recall_Weighted', 'F1_Weighted',
                          'AUC_Macro', 'AUC_Micro'] + 
                         [f'Precision_{class_name}' for class_name in self.class_names] +
                         [f'Recall_{class_name}' for class_name in self.class_names] +
                         [f'F1_{class_name}' for class_name in self.class_names] +
                         [f'AUC_{class_name}' for class_name in self.class_names])
        
        for model_name in self.models:
            model_data = results_df[results_df['Model'] == model_name.upper()]
            
            # Mean row
            mean_row = {
                'Model': model_name.upper(),
                'Fold': 'MEAN',
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
                'Best_Params': 'N/A'
            }
            for metric in metric_columns:
                if metric in model_data.columns:
                    std_row[metric] = round(model_data[metric].std(), 4)
            summary_data.append(std_row)
        
        # Combine all results
        summary_df = pd.DataFrame(summary_data)
        final_df = pd.concat([results_df, summary_df], ignore_index=True)
        
        return final_df
    
    def get_per_class_summary(self) -> pd.DataFrame:
        """
        Create a focused summary DataFrame showing per-class performance for each model.
        This is especially useful for analyzing void detection performance.
        """
        summary_data = []
        
        for model_name in self.models:
            if self.outer_scores[model_name] and self.auc_scores[model_name]:
                scores_df = pd.DataFrame(self.outer_scores[model_name])
                auc_df = pd.DataFrame(self.auc_scores[model_name])
                
                for class_name in self.class_names:
                    precision_col = f'precision_{class_name}'
                    recall_col = f'recall_{class_name}'
                    f1_col = f'f1_{class_name}'
                    auc_col = f'auc_{class_name}'
                    
                    if recall_col in scores_df.columns and auc_col in auc_df.columns:
                        row = {
                            'Model': model_name.upper(),
                            'Class': class_name,
                            'Precision_Mean': round(scores_df[precision_col].mean(), 4),
                            'Precision_Std': round(scores_df[precision_col].std(), 4),
                            'Recall_Mean': round(scores_df[recall_col].mean(), 4),
                            'Recall_Std': round(scores_df[recall_col].std(), 4),
                            'F1_Mean': round(scores_df[f1_col].mean(), 4),
                            'F1_Std': round(scores_df[f1_col].std(), 4),
                            'AUC_Mean': round(auc_df[auc_col].mean(), 4),
                            'AUC_Std': round(auc_df[auc_col].std(), 4)
                        }
                        summary_data.append(row)
        
        return pd.DataFrame(summary_data)

    def get_auc_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame focused on AUC scores across all models and classes.
        """
        summary_data = []
        
        for model_name in self.models:
            if self.auc_scores[model_name]:
                auc_df = pd.DataFrame(self.auc_scores[model_name])
                
                # Add macro and micro AUC
                summary_data.append({
                    'Model': model_name.upper(),
                    'Metric_Type': 'Macro_AUC',
                    'Mean': round(auc_df['macro_auc'].mean(), 4),
                    'Std': round(auc_df['macro_auc'].std(), 4),
                    'Min': round(auc_df['macro_auc'].min(), 4),
                    'Max': round(auc_df['macro_auc'].max(), 4),
                    'Individual_Folds': [round(x, 4) for x in auc_df['macro_auc'].tolist()]
                })
                
                summary_data.append({
                    'Model': model_name.upper(),
                    'Metric_Type': 'Micro_AUC',
                    'Mean': round(auc_df['micro_auc'].mean(), 4),
                    'Std': round(auc_df['micro_auc'].std(), 4),
                    'Min': round(auc_df['micro_auc'].min(), 4),
                    'Max': round(auc_df['micro_auc'].max(), 4),
                    'Individual_Folds': [round(x, 4) for x in auc_df['micro_auc'].tolist()]
                })
                
                # Add per-class AUC
                for class_name in self.class_names:
                    auc_col = f'auc_{class_name}'
                    if auc_col in auc_df.columns:
                        summary_data.append({
                            'Model': model_name.upper(),
                            'Metric_Type': f'AUC_{class_name}',
                            'Mean': round(auc_df[auc_col].mean(), 4),
                            'Std': round(auc_df[auc_col].std(), 4),
                            'Min': round(auc_df[auc_col].min(), 4),
                            'Max': round(auc_df[auc_col].max(), 4),
                            'Individual_Folds': [round(x, 4) for x in auc_df[auc_col].tolist()]
                        })
        
        return pd.DataFrame(summary_data)

    def plot_roc_curves(self, model_name: str = None, save_path: str = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot ROC curves for all models and folds.
        
        Args:
            model_name: If specified, plot only for this model. Otherwise plot for all models.
            save_path: Path to save the plot. If None, displays the plot.
            figsize: Figure size as (width, height)
        """
        models_to_plot = [model_name] if model_name else self.models
        
        # Set up colors for classes
        colors = plt.cm.Set1(np.linspace(0, 1, self.n_classes))
        
        fig, axes = plt.subplots(1, len(models_to_plot), figsize=figsize)
        if len(models_to_plot) == 1:
            axes = [axes]
        
        for model_idx, model in enumerate(models_to_plot):
            if not self.roc_data[model]:
                continue
                
            ax = axes[model_idx]
            
            # Calculate mean ROC curves for each class
            for class_idx, class_name in enumerate(self.class_names):
                all_fpr = []
                all_tpr = []
                aucs = []
                
                for fold_data in self.roc_data[model]:
                    if class_name in fold_data['roc_auc'] and fold_data['roc_auc'][class_name] > 0:
                        all_fpr.append(fold_data['fpr'][class_name])
                        all_tpr.append(fold_data['tpr'][class_name])
                        aucs.append(fold_data['roc_auc'][class_name])
                
                if aucs:  # Only plot if we have valid data
                    # Interpolate all curves to a common FPR axis
                    mean_fpr = np.linspace(0, 1, 100)
                    tprs = []
                    
                    for fpr, tpr in zip(all_fpr, all_tpr):
                        tprs.append(np.interp(mean_fpr, fpr, tpr))
                    
                    mean_tpr = np.mean(tprs, axis=0)
                    mean_auc = np.mean(aucs)
                    std_auc = np.std(aucs)
                    
                    ax.plot(mean_fpr, mean_tpr, 
                           color=colors[class_idx], linewidth=2,
                           label=f'{class_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
                    
                    # Add confidence interval
                    std_tpr = np.std(tprs, axis=0)
                    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
                    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
                    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, 
                                   color=colors[class_idx], alpha=0.2)
            
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(f'{model.upper()}: ROC Curves\n(Mean ± Std across {self.n_outer_folds} folds)')
            ax.legend(loc="lower right", fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to: {save_path}")
        else:
            plt.show()

    def plot_auc_comparison(self, save_path: str = None, figsize: Tuple[int, int] = (12, 8)):
        """
        Plot AUC scores comparison across models and classes.
        """
        # Prepare data for plotting
        plot_data = []
        for model_name in self.models:
            if self.auc_scores[model_name]:
                auc_df = pd.DataFrame(self.auc_scores[model_name])
                
                # Add macro and micro AUC
                for fold_idx in range(len(auc_df)):
                    plot_data.append({
                        'Model': model_name.upper(),
                        'Class': 'Macro_AUC',
                        'AUC': auc_df.iloc[fold_idx]['macro_auc'],
                        'Fold': fold_idx + 1
                    })
                    plot_data.append({
                        'Model': model_name.upper(),
                        'Class': 'Micro_AUC',
                        'AUC': auc_df.iloc[fold_idx]['micro_auc'],
                        'Fold': fold_idx + 1
                    })
                
                # Add per-class AUC
                for class_name in self.class_names:
                    auc_col = f'auc_{class_name}'
                    if auc_col in auc_df.columns:
                        for fold_idx in range(len(auc_df)):
                            plot_data.append({
                                'Model': model_name.upper(),
                                'Class': class_name,
                                'AUC': auc_df.iloc[fold_idx][auc_col],
                                'Fold': fold_idx + 1
                            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Macro AUC comparison
        macro_data = plot_df[plot_df['Class'] == 'Macro_AUC']
        for model in self.models:
            model_data = macro_data[macro_data['Model'] == model.upper()]
            axes[0, 0].plot(model_data['Fold'], model_data['AUC'], 'o-', 
                           label=model.upper(), linewidth=2, markersize=6)
        axes[0, 0].set_title('Macro AUC Across Folds')
        axes[0, 0].set_xlabel('Fold')
        axes[0, 0].set_ylabel('Macro AUC')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 1])
        
        # Plot 2: Per-class AUC for each model (boxplot)
        class_data = plot_df[~plot_df['Class'].isin(['Macro_AUC', 'Micro_AUC'])]
        if not class_data.empty:
            import seaborn as sns
            sns.boxplot(data=class_data, x='Class', y='AUC', hue='Model', ax=axes[0, 1])
            axes[0, 1].set_title('Per-Class AUC Distribution')
            axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Plot 3: Mean AUC comparison (bar plot)
        mean_aucs = []
        for model in self.models:
            if self.auc_scores[model]:
                auc_df = pd.DataFrame(self.auc_scores[model])
                mean_aucs.append({
                    'Model': model.upper(),
                    'Macro_AUC': auc_df['macro_auc'].mean(),
                    'Micro_AUC': auc_df['micro_auc'].mean()
                })
        
        mean_df = pd.DataFrame(mean_aucs)
        x = np.arange(len(mean_df))
        width = 0.35
        axes[1, 0].bar(x - width/2, mean_df['Macro_AUC'], width, label='Macro AUC')
        axes[1, 0].bar(x + width/2, mean_df['Micro_AUC'], width, label='Micro AUC')
        axes[1, 0].set_xlabel('Model')
        axes[1, 0].set_ylabel('Mean AUC')
        axes[1, 0].set_title('Mean AUC Comparison')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(mean_df['Model'])
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1])
        
        # Plot 4: AUC standard deviation (stability)
        std_aucs = []
        for model in self.models:
            if self.auc_scores[model]:
                auc_df = pd.DataFrame(self.auc_scores[model])
                std_aucs.append({
                    'Model': model.upper(),
                    'Macro_AUC_Std': auc_df['macro_auc'].std(),
                    'Micro_AUC_Std': auc_df['micro_auc'].std()
                })
        
        std_df = pd.DataFrame(std_aucs)
        x = np.arange(len(std_df))
        axes[1, 1].bar(x - width/2, std_df['Macro_AUC_Std'], width, label='Macro AUC Std')
        axes[1, 1].bar(x + width/2, std_df['Micro_AUC_Std'], width, label='Micro AUC Std')
        axes[1, 1].set_xlabel('Model')
        axes[1, 1].set_ylabel('AUC Standard Deviation')
        axes[1, 1].set_title('AUC Stability (Lower = More Stable)')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(std_df['Model'])
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"AUC comparison plots saved to: {save_path}")
        else:
            plt.show()