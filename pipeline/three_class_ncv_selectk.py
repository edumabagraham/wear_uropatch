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
import seaborn as sns
warnings.filterwarnings('ignore')



class NestedCVOptimizer:
    def __init__(self, X, y, groups, n_outer_folds=3, n_inner_folds=2,
                n_trials=50, random_state=42):
        """
        Nested cross-validation optimizer for 3-class classification
        with RandomForest, XGBoost, and DecisionTree including AUC analysis.
        
        Uses SelectKBest for feature selection.
        Each model gets its own hyperparameter optimization in each outer fold.
        
        Expected classes: 'pre-void', 'void', 'post-void'
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
        
        # Verify we have exactly 3 classes
        if self.n_classes != 3:
            raise ValueError(f"Expected 3 classes, but found {self.n_classes}: {self.class_names}")
        
        print(f"Detected classes (alphabetical order): {self.class_names}")
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, self.label_encoder.transform(self.label_encoder.classes_)))}")

        # Cross-validation splitters
        self.outer_cv = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
        self.inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)

        # Results storage
        self.models = ['rf', 'xgb', 'dt']
        self.outer_scores = {model: [] for model in self.models}
        self.best_params_per_fold = {model: [] for model in self.models}
        self.optimization_histories = {model: [] for model in self.models}
        
        self.confusion_matrices = {model: [] for model in self.models}
        
        # AUC storage
        self.auc_scores = {model: [] for model in self.models}  # Store AUC scores for each fold
        self.roc_data = {model: [] for model in self.models}  # Store ROC curves for plotting

    def get_model_search_space(self, model_name: str, trial) -> Dict[str, Any]:
        """Define model-specific hyperparameter search space with SelectKBest."""
        # SelectKBest parameters
        n_features = self.X.shape[1]
        
        # Choose number of features to select (k parameter)
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
        
        # Use f_classif for multi-class feature selection
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
                                objective='multi:softprob')  # Multi-class objective
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

    def calculate_roc_curves(self, y_true, y_proba, fold_idx: int, model_name: str) -> Dict[str, Any]:
        """
        Calculate ROC curves for visualization.
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
        """Calculate comprehensive classification metrics including per-class metrics."""
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

                    # Calculate negative multi-class log loss (maximize this = minimize log loss)
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
    
    # 2. Add method to calculate and store confusion matrices:
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

    def run_nested_cv(self) -> Dict[str, Any]:
        """Run the complete nested CV optimization for all models."""
        print("Starting 3-Class Nested Cross-Validation with SelectKBest feature selection...")
        print(f"Classes: {self.class_names}")
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
                    
                    # Calculate and store confusion matrix
                    cm_data = self.calculate_confusion_matrix(y_test_outer, y_pred, fold_idx, model_name)
                    self.confusion_matrices[model_name].append(cm_data)
                    
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
                    print(f"{model_name} test F1 macro: {fold_metrics['f1_macro']:.4f}")
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
                    
                    default_metrics = {
                        'accuracy': 0.0, 'precision_macro': 0.0, 'recall_macro': 0.0, 'f1_macro': 0.0,
                        'precision_weighted': 0.0, 'recall_weighted': 0.0, 'f1_weighted': 0.0
                    }
                    for class_name in self.class_names:
                        default_metrics[f'precision_{class_name}'] = 0.0
                        default_metrics[f'recall_{class_name}'] = 0.0
                        default_metrics[f'f1_{class_name}'] = 0.0
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
        print("3-CLASS NESTED CROSS-VALIDATION RESULTS SUMMARY")
        print(f"Classes: {self.class_names}")
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
                # Show key multi-class classification metrics
                key_metrics = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
                for metric in key_metrics:
                    if metric in scores_df.columns:
                        mean_val = scores_df[metric].mean()
                        std_val = scores_df[metric].std()
                        print(f"{metric:18}: {mean_val:.4f} ± {std_val:.4f}")
                
                # Print AUC scores
                print(f"{'macro_auc':18}: {auc_df['macro_auc'].mean():.4f} ± {auc_df['macro_auc'].std():.4f}")
                print(f"{'micro_auc':18}: {auc_df['micro_auc'].mean():.4f} ± {auc_df['micro_auc'].std():.4f}")
                
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
                print(f"Individual fold F1 (macro): {[f'{score:.4f}' for score in scores_df['f1_macro'].tolist()]}")
                print(f"Individual fold macro AUCs: {[f'{score:.4f}' for score in auc_df['macro_auc'].tolist()]}")

        # Find best model based on F1 macro score
        if summaries:
            best_model = max(summaries.keys(), 
                        key=lambda x: summaries[x]['mean_scores'].get('f1_macro', 0))
            best_f1_macro = summaries[best_model]['mean_scores']['f1_macro']
            best_accuracy = summaries[best_model]['mean_scores']['accuracy']
            best_macro_auc = summaries[best_model]['mean_auc_scores']['macro_auc']
            
            print(f"\n{'='*80}")
            print(f"BEST MODEL: {best_model.upper()}")
            print(f"F1 Score (macro): {best_f1_macro:.4f}")
            print(f"Accuracy: {best_accuracy:.4f}")
            print(f"Macro AUC: {best_macro_auc:.4f}")
            print(f"{'='*80}")
            
            summaries['best_model'] = best_model
            summaries['best_f1_macro'] = best_f1_macro
            summaries['best_accuracy'] = best_accuracy
            summaries['best_macro_auc'] = best_macro_auc

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
        Get the macro AUC scores for each model across all outer folds.
        """
        fold_aucs = {}
        for model_name in self.models:
            aucs = [fold_scores['macro_auc'] for fold_scores in self.auc_scores[model_name]]
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
        Create a comprehensive DataFrame with fold results including 3-class classification metrics
        and AUC scores for all classes.
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
                    
                    # Feature selection info
                    'Features_K': best_params.get('selector__k', 'N/A'),
                    'Score_Function': best_params.get('selector__score_func', 'N/A'),
                    
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

    def get_per_class_summary(self) -> pd.DataFrame:
        """
        Create a focused summary DataFrame showing per-class performance for each model.
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

    def plot_roc_curves(self, save_path: str = None, figsize: Tuple[int, int] = (15, 10)):
        """
        Plot ROC curves for all models and classes across folds.
        
        Args:
            save_path: Path to save the plot. If None, displays the plot.
            figsize: Figure size as (width, height)
        """
        # Set up colors for classes
        colors = plt.cm.Set1(np.linspace(0, 1, self.n_classes))
        
        fig, axes = plt.subplots(1, len(self.models), figsize=figsize)
        if len(self.models) == 1:
            axes = [axes]
        
        for model_idx, model in enumerate(self.models):
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
            if not model_data.empty:
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
        
        if mean_aucs:
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
        if mean_aucs:
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
        
        # Plot 3: Score function usage (should all be f_classif for 3-class)
        if 'Score_Function' in feature_summary.columns:
            score_func_counts = feature_summary['Score_Function'].value_counts()
            axes[1, 0].pie(score_func_counts.values, labels=score_func_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Score Function Usage Distribution')
        
        # Plot 4: Feature selection vs performance correlation
        if self.auc_scores:
            correlation_data = []
            for model in self.models:
                model_features = feature_summary[feature_summary['Model'] == model.upper()]
                model_aucs = [fold['macro_auc'] for fold in self.auc_scores[model]] if self.auc_scores[model] else []
                
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
                
                axes[1, 1].set_title('Feature Proportion vs Macro AUC Performance')
                axes[1, 1].set_xlabel('Proportion of Features Selected')
                axes[1, 1].set_ylabel('Macro AUC Score')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature selection analysis saved to: {save_path}")
        else:
            plt.show()
            
    # 4. Add method to get confusion matrix summary:
    def get_confusion_matrices_summary(self) -> Dict[str, Any]:
        """
        Get summary of confusion matrices across all folds and models.
        """
        summary = {}
        
        for model_name in self.models:
            if self.confusion_matrices[model_name]:
                model_cms = []
                
                for cm_data in self.confusion_matrices[model_name]:
                    model_cms.append({
                        'fold': cm_data['fold'],
                        'confusion_matrix': cm_data['confusion_matrix']
                    })
                
                # Calculate mean confusion matrix across folds
                cms = [cm_data['confusion_matrix'] for cm_data in self.confusion_matrices[model_name]]
                if cms:
                    mean_cm = np.mean(cms, axis=0)
                    std_cm = np.std(cms, axis=0)
                    
                    summary[model_name] = {
                        'individual_cms': model_cms,
                        'mean_cm': mean_cm,
                        'std_cm': std_cm,
                        'sum_cm': np.sum(cms, axis=0)  # Aggregated across all folds
                    }
        
        return summary

    # 5. Add plotting method for confusion matrices:
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
        else:
            # For binary implementation, use positive/negative labels
            class_labels = [f'Not_{self.positive_class}', self.positive_class]
        
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

    # 6. Add method to get detailed confusion matrix DataFrame:
    def get_confusion_matrix_dataframe(self) -> pd.DataFrame:
        """
        Create a comprehensive DataFrame with confusion matrix data for all models and folds.
        """
        cm_data = []
        
        # Determine class labels based on implementation type
        if hasattr(self, 'class_names'):
            class_labels = self.class_names
            n_classes = len(self.class_names)
        else:
            class_labels = [f'Not_{self.positive_class}', self.positive_class]
            n_classes = 2
        
        for model_name in self.models:
            for cm_info in self.confusion_matrices[model_name]:
                fold_idx = cm_info['fold']
                cm = cm_info['confusion_matrix']
                
                # Flatten confusion matrix into row format
                row = {
                    'Model': model_name.upper(),
                    'Fold': fold_idx + 1
                }
                
                # Add each cell of confusion matrix
                for i in range(n_classes):
                    for j in range(n_classes):
                        if i < cm.shape[0] and j < cm.shape[1]:
                            row[f'CM_{class_labels[i]}_pred_{class_labels[j]}'] = cm[i, j]
                        else:
                            row[f'CM_{class_labels[i]}_pred_{class_labels[j]}'] = 0
                
                # Calculate derived metrics from confusion matrix
                if n_classes == 2:  # Binary classification
                    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
                    row.update({
                        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
                        'Sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
                        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
                        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0
                    })
                
                cm_data.append(row)
        
        return pd.DataFrame(cm_data)

    # 7. Add method to plot per-fold confusion matrices:
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