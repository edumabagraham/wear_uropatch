from hmmlearn import hmm
import numpy as np
import pandas as pd


class ApplyHMM:
    def __init__(self, train_df, test_df, predictions_prob_df):
        self.train_df = train_df
        self.predictions_prob_df = predictions_prob_df
        self.test_df = test_df
        
        
    # STEP 1: Learn 3-class transition matrix from training ground truth
    # =========================================================================
    def calculate_transition_and_start_probabilities(self, state_col='label', 
                                    id_col='experiment_id', time_col='center_time'):
            
        states = ['pre-void', 'void', 'post-void']
        state_to_idx = {state: i for i, state in enumerate(states)}
        transition_counts = np.zeros((3, 3))
            
        for exp_id, group in self.train_df.groupby(id_col):
            group = group.sort_values(time_col)
            states_sequence = group[state_col].values
                
            for i in range(len(states_sequence) - 1):
                current_idx = state_to_idx[states_sequence[i]]
                next_idx = state_to_idx[states_sequence[i + 1]]
                transition_counts[current_idx, next_idx] += 1
            
        # Convert to probabilities
        transmat = np.zeros((3, 3))
        for i in range(3):
            row_sum = transition_counts[i, :].sum()
            if row_sum > 0:
                transmat[i, :] = transition_counts[i, :] / row_sum
            else:
                transmat[i, i] = 1.0
                
        # Start probabilities - should heavily favor pre-void
        startprob = np.zeros(3)
        for exp_id, group in self.train_df.groupby(id_col):
            first_state = group.sort_values(time_col).iloc[0][state_col]
            startprob[state_to_idx[first_state]] += 1
        startprob = startprob / startprob.sum()
            
        return transmat, startprob
    
    def improved_void_detection(self, void_prob_col='void', nonvoid_prob_col='non-void', 
                        id_col='experiment_id', n_states=3, seeds=range(50)):
        """
        Improved void detection using full probability distributions from binary classifier
        
        predictions_prob_df: 
        """
        # Get transition and start probabilities from training data        
        transmat, startprob = self.calculate_transition_and_start_probabilities()
        
        # Prepare observations using BOTH probabilities
        all_obs = []
        lengths = []
        
        for exp_id, group in self.predictions_prob_df.groupby(id_col):
            group = group.sort_values('center_time')
            # Use both void and non-void probabilities as 2D observations
            void_probs = group[void_prob_col].to_numpy()
            nonvoid_probs = group[nonvoid_prob_col].to_numpy()
            
            # Stack as 2D observations: [non-void_prob, void_prob]
            observations = np.column_stack([nonvoid_probs, void_probs])
            all_obs.append(observations)
            lengths.append(len(observations))
        
        all_obs = np.vstack(all_obs)
        
        best_score = -np.inf
        best_model = None
        best_seed = None
        
        # Try multiple seeds
        for seed in seeds:
            try:
                # Use GaussianHMM for continuous probability observations
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type="diag",  # Allow correlation between void/nonvoid probs
                    init_params="mc",  # Learn means and covariances, fix transitions
                    n_iter=50,
                    tol=1e-3,
                    random_state=seed
                )                

                
                model.startprob_ = startprob
                model.transmat_ = transmat
                
                model.fit(all_obs, lengths)
                score = model.score(all_obs, lengths)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_seed = seed
                    
            except Exception as e:
                print(f"Seed {seed} failed: {e}")
                continue
        
        if best_model is None:
            raise ValueError("All seeds failed to converge!")
        
        # Decode using the best model
        state_map = {0: 'pre-void', 1: 'void', 2: 'post-void'} # this is different from the label encoding.
        all_results = []
        
        for exp_id, group in self.predictions_prob_df.groupby(id_col):
            void_probs = group[void_prob_col].to_numpy()
            nonvoid_probs = group[nonvoid_prob_col].to_numpy()
            observations = np.column_stack([nonvoid_probs, void_probs])
            
            hidden_states = best_model.decode(observations, algorithm="viterbi")[1]
            
            group_result = group.copy()
            group_result['predicted_state'] = [state_map[s] for s in hidden_states]
            all_results.append(group_result)
        
        return pd.concat(all_results), best_model, best_score, best_seed, transmat, startprob
    
    def analyze_gaussian_emissions(self, model):
        """
        Analyze what the Gaussian HMM learned about each state
        """
        print("=== Learned Gaussian Emissions ===")
        
        state_names = ['pre-void', 'void', 'post-void']
        
        for i, state in enumerate(state_names):
            mean = model.means_[i]
            cov = model.covars_[i]
            
            print(f"\n{state.upper()}:")
            print(f"  Mean probabilities: [non-void: {mean[0]:.3f}, void: {mean[1]:.3f}]")
            print(f"  Covariance matrix:")
            print(f"    {cov}")
            
            # Interpret the learned pattern
            if mean[1] > mean[0]:
                tendency = "VOID events (urination likely)"
            else:
                tendency = "NON-VOID events (normal state)"
            print(f"  → This state tends to emit: {tendency}")