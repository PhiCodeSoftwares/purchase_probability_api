import numpy as np
from hmmlearn import hmm
import pickle

observations = {
    "en": [
        "High Need", "Low Need", "Low Money", "High Money",
        "High Satisfaction", "Low Satisfaction", "With Card Limit",
        "No Card Limit", "Low Expenses", "High Expenses",
        "Correct Last Purchase", "Wrong Last Purchase",

        # (External/Contextual Factors)
        "Active Promotion", "No Promotion",
        "Upcoming Holiday", "No Upcoming Holiday",
        "Relevant News", "No Relevant News",
        "Social Media Influence", "No Social Media Influence",

        # (User History)
        "Recent Similar Purchase", "No Recent Similar Purchase",
        "Browsed Product Page", "Did Not Browse Product Page",
        "Viewed Reviews", "Did Not View Reviews",

        # (Product Characteristics)
        "High Average Rating", "Low Average Rating",
    ],
    "pt-br": [
        "Alta Necessidade", "Baixa Necessidade", "Pouco Dinheiro", "Muito Dinheiro",
        "Alta Satisfação", "Baixa Satisfação", "Com Limite no Cartão",
        "Sem Limite no Cartão", "Baixas Despesas", "Altas Despesas",
        "Última Compra Correta", "Última Compra Errada",

        # (Fatores Externos/Contextuais)
        "Promoção Ativa", "Sem Promoção",
        "Feriado Próximo", "Sem Feriado Próximo",
        "Notícias Relevantes", "Sem Notícias Relevantes",
        "Influência de Mídia Social", "Sem Influência de Mídia Social",

        # (Histórico do Usuário)
        "Compra Recente Similar", "Sem Compra Recente Similar",
        "Visualizou Página do Produto", "Não Visualizou Página do Produto",
        "Visualizou Avaliações", "Não Visualizou Avaliações",

        # (Características do Produto)
        "Alta Classificação Média", "Baixa Classificação Média"
    ]
}

# HMM Model Class
class HMMModel:
    def __init__(self):
        self.model = None
        
        self.run()
    
    def run(self):
        # Define the state space
        self.states = ["Purchase", "No Purchase"]
        self.n_states = len(self.states)

        self.n_observations = len(observations["en"])

        # Define the initial state distribution
        self.state_probability = np.array([0.4, 0.6])

        # Define the state transition probabilities
        self.transition_probability = np.array([
            [0.7, 0.3],
            [0.3, 0.7]
        ])

        emission_probs = np.array([
            [
                60, 10, 10, 50, 40, 15, 20, 25, 35, 45, 25, 5, 
                25, 15, 15, 10, 20, 10, 25, 10, 
                20, 5, 15, 5, 10, 5, 25, 5
            ],
            [
                10, 60, 60, 10, 15, 40, 10, 25, 20, 45, 5, 30, 
                10, 20, 10, 20, 10, 20, 10, 20, 
                15, 10, 5, 20, 5, 20, 10, 25
            ]
        ])
        self.emission_probability = emission_probs / emission_probs.sum(axis=1, keepdims=True)

        # Create the HMM model
        self.model = hmm.CategoricalHMM(n_components=self.n_states)
        self.model.startprob_ = self.state_probability
        self.model.transmat_ = self.transition_probability
        self.model.emissionprob_ = self.emission_probability

    def get_observations(self, language):
        return observations[language]

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
    
    def save_model(self):
        with open("./model/hmm_model.pkl", "wb") as f:
            pickle.dump(self, f)