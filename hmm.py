import numpy as np
from hmmlearn import hmm
import pickle

# HMM Model Class
class HMMModel:
    def __init__(self):
        # Define the state space
        self.states = ["Purchase", "No Purchase"]
        self.n_states = len(self.states)

        # Define the observation space
        self.observations = [
            "High Need", "Low Need", "Low Money", "High Money",
            "High Satisfaction", "Low Satisfaction", "With Card Limit",
            "No Card Limit", "Low Expenses", "High Expenses",
            "Correct Last Purchase", "Wrong Last Purchase"
        ]
        self.n_observations = len(self.observations)

        # Define the initial state distribution
        self.state_probability = np.array([0.4, 0.6])

        # Define the state transition probabilities
        self.transition_probability = np.array([
            [0.7, 0.3],
            [0.3, 0.7]
        ])

        # Define the emission probabilities
        self.emission_probability = np.array([
            [0.2, 0.1, 0.05, 0.1, 0.15, 0.05, 0.1, 0.05, 0.1, 0.05, 0.03, 0.02],
            [0.05, 0.2, 0.1, 0.05, 0.05, 0.15, 0.05, 0.05, 0.05, 0.1, 0.05, 0.1]
        ])

        # Create the HMM model
        self.model = hmm.CategoricalHMM(n_components=self.n_states)
        self.model.startprob_ = self.state_probability
        self.model.transmat_ = self.transition_probability
        self.model.emissionprob_ = self.emission_probability

    def get_observations(self):
        return self.observations

    def predict_probabilities(self, observation_sequence):
        # Predict the probabilities for each hidden state given the observation sequence
        state_probabilities = self.model.predict_proba(observation_sequence)

        # Calculate the total probabilities for "Purchase" and "No Purchase"
        total_prob_purchase = np.sum(state_probabilities[:, 0])
        total_prob_no_purchase = np.sum(state_probabilities[:, 1])

        # Normalize to get the aggregated probabilities
        total_prob = total_prob_purchase + total_prob_no_purchase
        prob_purchase_aggregated = (total_prob_purchase / total_prob) * 100
        prob_no_purchase_aggregated = (total_prob_no_purchase / total_prob) * 100

        return prob_purchase_aggregated, prob_no_purchase_aggregated
    
# Save the HMM model to a pickle file
hmm_model = HMMModel()
with open("hmm_model.pkl", "wb") as f:
    pickle.dump(hmm_model, f)