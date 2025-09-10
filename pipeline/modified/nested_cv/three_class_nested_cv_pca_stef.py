import numpy as np
import optuna
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           roc_curve, auc, roc_auc_score, log_loss, balanced_accuracy_score)
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from typing import Dict, List, Any, Tuple, Optional
warnings.filterwarnings('ignore')


class ImprovedNestedCVOptimizer:
    def __init__(self, X, y, groups,
                 n_outer_folds=5,
                 n_inner_folds=5,
                 n_trials=200,
                 optimization_metric='balanced_accuracy',
                 include_baseline=True,
                 random_state=42):
        """
        Improved nested cross-validation optimizer with flexible preprocessing and optimization.
       
        Parameters:
        -----------
        optimization_metric : str
            Metric to optimize during hyperparameter search. Options:
            'accuracy', 'balanced_accuracy', 'f1_macro', 'log_loss', 'roc_auc'
        include_baseline : bool
            Whether to include baseline models without preprocessing
        """
        self.X = X
        self.y = y
        self.groups = groups
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.n_trials = n_trials
        self.optimization_metric = optimization_metric
        self.include_baseline = include_baseline
        self.random_state = random_state
       
        # Label encoding
        self.label_encoder = LabelEncoder()
        self.y_encoded = self.label_encoder.fit_transform(self.y)
       
        # Class information
        self.class_names = sorted(np.unique(self.y))
        self.n_classes = len(self.class_names)
        self.n_features = X.shape[1]
       
        # Calculate class weights for imbalanced data
        self.class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(self.y_encoded),
            y=self.y_encoded
        )
        self.class_weight_dict = dict(zip(range(len(self.class_weights)), self.class_weights))
       
        print(f"Dataset info:")
        print(f"- Classes: {self.class_names}")
        print(f"- Class distribution: {np.bincount(self.y_encoded)}")
        print(f"- Features: {self.n_features}")
        print(f"- Samples: {len(self.y)}")
        print(f"- Class weights: {self.class_weight_dict}")
        print(f"- Optimization metric: {optimization_metric}")
        print(f"- CV strategy: {n_outer_folds}x{n_inner_folds} nested CV")
       
        # Cross-validation splitters
        self.outer_cv = StratifiedGroupKFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
        self.inner_cv = StratifiedGroupKFold(n_splits=n_inner_folds, shuffle=True, random_state=random_state)
       
        # Model configurations
        self.models = ['rf', 'xgb', 'dt']
        if include_baseline:
            self.models.extend(['rf_baseline', 'xgb_baseline'])  # Baseline models without preprocessing
       
        # Results storage
        self.outer_scores = {model: [] for model in self.models}
        self.best_params_per_fold = {model: [] for model in self.models}
        self.optimization_histories = {model: [] for model in self.models}
        self.auc_scores = {model: [] for model in self.models}
        self.roc_data = {model: [] for model in self.models}
       
    def get_preprocessing_pipeline(self, trial, model_name: str) -> Tuple[List, Dict]:
        """
        Create preprocessing pipeline based on trial suggestions.
        Returns pipeline steps and parameter dict.
        """
        steps = []
        params = {}
       
        # Skip preprocessing for baseline models
        if 'baseline' in model_name:
            return steps, params
       
        # Step 1: Scaling (always applied before dimensionality reduction)
        scaler_type = trial.suggest_categorical('scaler', ['standard', 'robust', 'none'])
        if scaler_type == 'standard':
            steps.append(('scaler', StandardScaler()))
        elif scaler_type == 'robust':
            steps.append(('scaler', RobustScaler()))
       
        # Step 2: Dimensionality reduction (optional)
        dim_reduction = trial.suggest_categorical('dim_reduction', ['none', 'pca', 'lda', 'select_k'])
       
        if dim_reduction == 'pca':
            # PCA with variance ratio or fixed components
            pca_type = trial.suggest_categorical('pca_type', ['variance', 'fixed'])
            if pca_type == 'variance':
                n_components = trial.suggest_categorical('pca__n_components',
                                                        [0.85, 0.90, 0.95, 0.99])
            else:
                max_components = min(self.n_features - 1, self.X.shape[0] - 1, 100)
                n_components = trial.suggest_int('pca__n_components', 5, max_components)
           
            steps.append(('dim_reduce', PCA(n_components=n_components, random_state=self.random_state)))
            params['dim_reduce__n_components'] = n_components
           
        elif dim_reduction == 'lda':
            # LDA for supervised dimensionality reduction
            max_components = min(self.n_classes - 1, self.n_features)
            n_components = trial.suggest_int('lda__n_components', 1, max_components)
            steps.append(('dim_reduce', LinearDiscriminantAnalysis(n_components=n_components)))
            params['dim_reduce__n_components'] = n_components
           
        elif dim_reduction == 'select_k':
            # Feature selection using mutual information
            k = trial.suggest_int('select_k__k',
                                min(5, self.n_features),
                                self.n_features)
            steps.append(('dim_reduce', SelectKBest(mutual_info_classif, k=k)))
            params['dim_reduce__k'] = k
       
        return steps, params
   
    def get_model_search_space(self, model_name: str, trial) -> Tuple[Any, Dict[str, Any]]:
        """
        Define model-specific hyperparameter search space.
        Returns the classifier instance and parameters dict.
        """
        params = {}
        base_model = model_name.replace('_baseline', '')
       
        if base_model == 'rf':
            params.update({
                'clf__n_estimators': trial.suggest_int('clf__n_estimators', 100, 1000),
                'clf__max_depth': trial.suggest_categorical('clf__max_depth',
                                                        [None, 5, 10, 15, 20, 30, 50]),
                'clf__min_samples_split': trial.suggest_int('clf__min_samples_split', 2, 20),
                'clf__min_samples_leaf': trial.suggest_int('clf__min_samples_leaf', 1, 10),
                'clf__max_features': trial.suggest_categorical('clf__max_features',
                                                            ['sqrt', 'log2', 0.5, 0.8, None]),
                'clf__bootstrap': trial.suggest_categorical('clf__bootstrap', [True, False]),
                'clf__min_impurity_decrease': trial.suggest_float('clf__min_impurity_decrease',
                                                                0.0, 0.01),
            })
           
            # Class weight handling
            use_class_weight = trial.suggest_categorical('clf__use_class_weight', [True, False])
            if use_class_weight:
                clf = RandomForestClassifier(
                    class_weight='balanced',
                    random_state=self.random_state,
                    n_jobs=-1
                )
            else:
                clf = RandomForestClassifier(
                    random_state=self.random_state,
                    n_jobs=-1
                )
           
        elif base_model == 'xgb':
            # Calculate scale_pos_weight for imbalanced data
            scale_pos_weight = self.class_weights.max() / self.class_weights.min()
           
            params.update({
                'clf__n_estimators': trial.suggest_int('clf__n_estimators', 100, 1500),
                'clf__max_depth': trial.suggest_categorical('clf__max_depth',
                                                        [3, 5, 7, 10, 15, 20, None]),
                'clf__learning_rate': trial.suggest_float('clf__learning_rate', 0.001, 0.5, log=True),
                'clf__subsample': trial.suggest_float('clf__subsample', 0.5, 1.0),
                'clf__colsample_bytree': trial.suggest_float('clf__colsample_bytree', 0.5, 1.0),
                'clf__colsample_bylevel': trial.suggest_float('clf__colsample_bylevel', 0.5, 1.0),
                'clf__min_child_weight': trial.suggest_int('clf__min_child_weight', 1, 20),
                'clf__gamma': trial.suggest_float('clf__gamma', 0, 10),
                'clf__reg_alpha': trial.suggest_float('clf__reg_alpha', 0, 5),
                'clf__reg_lambda': trial.suggest_float('clf__reg_lambda', 0, 5),
            })
           
            # Handle scale_pos_weight for imbalanced data
            use_scale_weight = trial.suggest_categorical('clf__use_scale_weight', [True, False])
            if use_scale_weight and self.n_classes == 2:  # Only for binary classification
                params['clf__scale_pos_weight'] = scale_pos_weight
           
            clf = XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss',
                verbosity=0,
                use_label_encoder=False
            )
           
        elif base_model == 'dt':
            params.update({
                'clf__criterion': trial.suggest_categorical('clf__criterion', ['gini', 'entropy']),
                'clf__max_depth': trial.suggest_categorical('clf__max_depth',
                                                           [None, 3, 5, 10, 15, 20, 30]),
                'clf__min_samples_split': trial.suggest_int('clf__min_samples_split', 2, 50),
                'clf__min_samples_leaf': trial.suggest_int('clf__min_samples_leaf', 1, 20),
                'clf__max_features': trial.suggest_categorical('clf__max_features',
                                                              ['sqrt', 'log2', 0.5, 0.8, None]),
                'clf__splitter': trial.suggest_categorical('clf__splitter', ['best', 'random']),
                'clf__min_impurity_decrease': trial.suggest_float('clf__min_impurity_decrease',
                                                                 0.0, 0.05),
            })
           
            # Class weight handling
            use_class_weight = trial.suggest_categorical('clf__use_class_weight', [True, False])
            if use_class_weight:
                clf = DecisionTreeClassifier(
                    class_weight='balanced',
                    random_state=self.random_state
                )
            else:
                clf = DecisionTreeClassifier(
                    random_state=self.random_state
                )
        else:
            raise ValueError(f"Unknown model: {model_name}")
       
        return clf, params
   
    def create_model(self, model_name: str, preprocessing_steps: List,
                    clf: Any, all_params: Dict[str, Any]):
        """
        Create a complete pipeline with preprocessing and classifier.
        """
        # Extract only the classifier parameters
        clf_params = {k.replace('clf__', ''): v
                     for k, v in all_params.items()
                     if k.startswith('clf__') and not k.startswith('clf__use')}
       
        # Set the parameters on the classifier
        clf.set_params(**clf_params)
       
        # Build the complete pipeline
        pipeline_steps = preprocessing_steps + [('clf', clf)]
        pipeline = Pipeline(pipeline_steps)
       
        return pipeline
   
    def calculate_optimization_score(self, y_true, y_pred, y_proba=None):
        """
        Calculate the optimization metric score.
        """
        if self.optimization_metric == 'accuracy':
            return accuracy_score(y_true, y_pred)
        elif self.optimization_metric == 'balanced_accuracy':
            return balanced_accuracy_score(y_true, y_pred)
        elif self.optimization_metric == 'f1_macro':
            return f1_score(y_true, y_pred, average='macro', zero_division=0)
        elif self.optimization_metric == 'log_loss':
            if y_proba is not None:
                return -log_loss(y_true, y_proba)  # Negative because we maximize
            else:
                return 0.0
        elif self.optimization_metric == 'roc_auc':
            if y_proba is not None and self.n_classes == 2:
                return roc_auc_score(y_true, y_proba[:, 1])
            elif y_proba is not None:
                # Multi-class AUC
                y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
                return roc_auc_score(y_true_bin, y_proba, multi_class='ovr', average='macro')
            else:
                return 0.0
        else:
            raise ValueError(f"Unknown optimization metric: {self.optimization_metric}")
   
    def optimize_model(self, model_name: str, X_train_outer, y_train_outer,
                    groups_train_outer) -> Tuple[Dict, optuna.Study]:
        """
        Optimize hyperparameters for a specific model using inner CV.
        """
        def objective(trial):
            try:
                # Get preprocessing pipeline
                preprocessing_steps, preprocessing_params = self.get_preprocessing_pipeline(trial, model_name)

                # Get model and its parameters
                clf, model_params = self.get_model_search_space(model_name, trial)

                # Combine all parameters
                all_params = {**preprocessing_params, **model_params}

                # Perform inner CV
                fold_scores = []
                for train_idx, val_idx in self.inner_cv.split(X_train_outer, y_train_outer, groups_train_outer):
                    X_train_inner = X_train_outer.iloc[train_idx]
                    X_val_inner = X_train_outer.iloc[val_idx]
                    y_train_inner = y_train_outer.iloc[train_idx]
                    y_val_inner = y_train_outer.iloc[val_idx]

                    # Create and train model
                    model = self.create_model(model_name, preprocessing_steps.copy(), clf, all_params)
                    model.fit(X_train_inner, y_train_inner)

                    # Evaluate
                    y_pred = model.predict(X_val_inner)
                    y_proba = model.predict_proba(X_val_inner) if hasattr(model, 'predict_proba') else None

                    score = self.calculate_optimization_score(y_val_inner, y_pred, y_proba)
                    fold_scores.append(score)

                return np.mean(fold_scores) if fold_scores else -1.0

            except Exception as e:
                print(f"Trial failed for {model_name}: {e}")
                return -1.0

        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=10)
        )
       
        # Run optimization with progress bar
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)
       
        return study.best_params, study
   
    def calculate_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics.
        """
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
                'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
                'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
                'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
           
            # Per-class metrics
            for i, class_name in enumerate(self.class_names):
                precision = precision_score(y_true == i, y_pred == i, zero_division=0)
                recall = recall_score(y_true == i, y_pred == i, zero_division=0)
                f1 = f1_score(y_true == i, y_pred == i, zero_division=0)
               
                metrics[f'precision_{class_name}'] = precision
                metrics[f'recall_{class_name}'] = recall
                metrics[f'f1_{class_name}'] = f1
           
            return metrics
           
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            return {'accuracy': 0.0, 'balanced_accuracy': 0.0}
   
    def calculate_auc_scores(self, y_true, y_proba, fold_idx: int, model_name: str) -> Dict[str, Any]:
        """
        Calculate AUC scores for multi-class classification.
        """
        auc_data = {
            'fold': fold_idx,
            'model': model_name,
            'macro_auc': 0.0,
            'micro_auc': 0.0
        }

        try:
            if self.n_classes == 2:
                # Binary classification
                auc_data['macro_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                auc_data['micro_auc'] = auc_data['macro_auc']
            else:
                # Multi-class
                y_true_bin = label_binarize(y_true, classes=range(self.n_classes))
                auc_data['macro_auc'] = roc_auc_score(y_true_bin, y_proba,
                                                    multi_class='ovr', average='macro')
                auc_data['micro_auc'] = roc_auc_score(y_true_bin, y_proba,
                                                    multi_class='ovr', average='micro')

            # Per-class AUC
            for i, class_name in enumerate(self.class_names):
                y_true_binary = (y_true == i).astype(int)
                if len(np.unique(y_true_binary)) > 1:  # Check if both classes present
                    auc_data[f'auc_{class_name}'] = roc_auc_score(y_true_binary, y_proba[:, i])
                else:
                    auc_data[f'auc_{class_name}'] = 0.0

        except Exception as e:
            print(f"Error calculating AUC: {e}")
            for class_name in self.class_names:
                auc_data[f'auc_{class_name}'] = 0.0

        return auc_data

    def run_nested_cv(self) -> Dict[str, Any]:
        """
        Run the complete nested CV optimization.
        """
        print(f"\nStarting Nested Cross-Validation ({self.n_outer_folds}x{self.n_inner_folds})...")
        print(f"Models to evaluate: {self.models}")
        print(f"Optimization metric: {self.optimization_metric}")
        print("="*80)
       
        for fold_idx, (train_idx, test_idx) in enumerate(self.outer_cv.split(self.X, self.y_encoded, self.groups)):
            print(f"\n{'='*60}")
            print(f"OUTER FOLD {fold_idx + 1}/{self.n_outer_folds}")
            print(f"{'='*60}")
           
            # Prepare data
            X_train_outer = self.X.iloc[train_idx]
            X_test_outer = self.X.iloc[test_idx]
            y_train_outer = pd.Series(self.y_encoded, index=self.X.index).iloc[train_idx]
            y_test_outer = pd.Series(self.y_encoded, index=self.X.index).iloc[test_idx]
            groups_train_outer = self.groups.iloc[train_idx] if hasattr(self.groups, 'iloc') else self.groups[train_idx]
           
            print(f"Train size: {len(X_train_outer)}, Test size: {len(X_test_outer)}")
            print(f"Test class distribution: {np.bincount(y_test_outer)}")
           
            # Optimize each model
            for model_name in self.models:
                print(f"\n--- {model_name.upper()} ---")
               
                try:
                    # Optimize
                    best_params, study = self.optimize_model(model_name, X_train_outer,
                                                            y_train_outer, groups_train_outer)
                   
                    print(f"Best score: {study.best_value:.4f}")
                    print(f"Best params summary: {len(best_params)} parameters optimized")
                   
                    # Store results
                    self.best_params_per_fold[model_name].append(best_params)
                    self.optimization_histories[model_name].append({
                        'best_value': study.best_value,
                        'n_trials': len(study.trials)
                    })
                   
                    # Train final model
                    preprocessing_steps, _ = self.get_preprocessing_pipeline(
                        optuna.trial.FixedTrial(best_params), model_name
                    )
                    clf, _ = self.get_model_search_space(model_name,
                                                        optuna.trial.FixedTrial(best_params))
                    final_model = self.create_model(model_name, preprocessing_steps, clf, best_params)
                    final_model.fit(X_train_outer, y_train_outer)
                   
                    # Evaluate
                    y_pred = final_model.predict(X_test_outer)
                    y_proba = final_model.predict_proba(X_test_outer) if hasattr(final_model, 'predict_proba') else None
                   
                    # Calculate metrics
                    fold_metrics = self.calculate_metrics(y_test_outer, y_pred)
                    self.outer_scores[model_name].append(fold_metrics)
                   
                    # Calculate AUC
                    if y_proba is not None:
                        auc_data = self.calculate_auc_scores(y_test_outer, y_proba, fold_idx, model_name)
                        self.auc_scores[model_name].append(auc_data)
                   
                    # Print key results
                    print(f"Test Accuracy: {fold_metrics['accuracy']:.4f}")
                    print(f"Test Balanced Accuracy: {fold_metrics['balanced_accuracy']:.4f}")
                    if y_proba is not None:
                        print(f"Test Macro AUC: {auc_data['macro_auc']:.4f}")
                   
                except Exception as e:
                    print(f"ERROR in {model_name}: {e}")
                    # Store empty results
                    self.best_params_per_fold[model_name].append({})
                    self.optimization_histories[model_name].append({'best_value': 0.0, 'n_trials': 0})
                    self.outer_scores[model_name].append({'accuracy': 0.0, 'balanced_accuracy': 0.0})
                    self.auc_scores[model_name].append({'macro_auc': 0.0, 'micro_auc': 0.0})
       
        return self._summarize_results()
   
    def _summarize_results(self) -> Dict[str, Any]:
        """
        Summarize and display final results.
        """
        print(f"\n{'='*80}")
        print("FINAL RESULTS SUMMARY")
        print(f"{'='*80}")
       
        summaries = {}
       
        for model_name in self.models:
            if self.outer_scores[model_name]:
                scores_df = pd.DataFrame(self.outer_scores[model_name])
               
                summary = {
                    'mean_scores': scores_df.mean().to_dict(),
                    'std_scores': scores_df.std().to_dict(),
                    'all_fold_scores': self.outer_scores[model_name],
                    'best_params_per_fold': self.best_params_per_fold[model_name]
                }
               
                if self.auc_scores[model_name]:
                    auc_df = pd.DataFrame(self.auc_scores[model_name])
                    summary['mean_auc_scores'] = auc_df.mean(numeric_only=True).to_dict()
                    summary['std_auc_scores'] = auc_df.std(numeric_only=True).to_dict()
               
                summaries[model_name] = summary
               
                # Print summary
                print(f"\n{model_name.upper()} Results:")
                print("-" * 40)
               
                key_metrics = ['accuracy', 'balanced_accuracy', 'f1_macro', 'recall_macro']
                for metric in key_metrics:
                    if metric in scores_df.columns:
                        mean_val = scores_df[metric].mean()
                        std_val = scores_df[metric].std()
                        print(f"{metric:20}: {mean_val:.4f} ± {std_val:.4f}")
               
                if self.auc_scores[model_name] and model_name in summaries:
                    if 'mean_auc_scores' in summaries[model_name]:
                        macro_auc = summaries[model_name]['mean_auc_scores'].get('macro_auc', 0)
                        macro_std = summaries[model_name].get('std_auc_scores', {}).get('macro_auc', 0)
                        print(f"{'macro_auc':20}: {macro_auc:.4f} ± {macro_std:.4f}")
               
                # Print fold results
                fold_accs = scores_df['accuracy'].tolist()
                print(f"Fold accuracies: {[f'{s:.4f}' for s in fold_accs]}")
       
        # Find best model
        if summaries:
            best_model = max(summaries.keys(),
                           key=lambda x: summaries[x]['mean_scores'].get('balanced_accuracy', 0))
            best_score = summaries[best_model]['mean_scores']['balanced_accuracy']
           
            print(f"\n{'='*80}")
            print(f"BEST MODEL: {best_model.upper()}")
            print(f"Balanced Accuracy: {best_score:.4f}")
            print(f"{'='*80}")
           
            summaries['best_model'] = best_model
            summaries['best_score'] = best_score
       
        return summaries
   
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Create a comprehensive results DataFrame.
        """
        results_data = []
       
        for model_name in self.models:
            for fold_idx in range(len(self.outer_scores[model_name])):
                fold_metrics = self.outer_scores[model_name][fold_idx]
               
                row = {
                    'Model': model_name.upper(),
                    'Fold': fold_idx + 1,
                    'Accuracy': round(fold_metrics.get('accuracy', 0.0), 4),
                    'Balanced_Accuracy': round(fold_metrics.get('balanced_accuracy', 0.0), 4),
                    'F1_Macro': round(fold_metrics.get('f1_macro', 0.0), 4),
                    'Recall_Macro': round(fold_metrics.get('recall_macro', 0.0), 4),
                    'Precision_Macro': round(fold_metrics.get('precision_macro', 0.0), 4)
                }
               
                # Add AUC if available
                if self.auc_scores[model_name] and fold_idx < len(self.auc_scores[model_name]):
                    auc_data = self.auc_scores[model_name][fold_idx]
                    row['AUC_Macro'] = round(auc_data.get('macro_auc', 0.0), 4)
               
                results_data.append(row)
       
        return pd.DataFrame(results_data)
