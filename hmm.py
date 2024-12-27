import numpy as np
from hmmlearn import hmm

class HMMModel:
    def __init__(self):
        # Definir o espaço de estados
        self.states = ["Compra", "Não Compra"]
        self.n_states = len(self.states)

        # Definir o espaço de observações
        self.observations = [
            "Necessidade Alta", "Necessidade Baixa", "Pouco Dinheiro", 
            "Muito Dinheiro", "Satisfação Alta", "Satisfação Baixa", 
            "Com Limite", "Sem Limite", "Poucos Gastos", "Muitos Gastos", 
            "Última Compra Correta", "Última Compra Errada"
        ]
        self.n_observations = len(self.observations)

        # Definir a distribuição inicial de estados
        self.state_probability = np.array([0.4, 0.6])

        # Definir as probabilidades de transição entre os estados
        self.transition_probability = np.array([
            [0.7, 0.3],
            [0.3, 0.7]
        ])

        # Definir as probabilidades de emissão
        self.emission_probability = np.array([
            [0.2, 0.1, 0.05, 0.1, 0.15, 0.05, 0.1, 0.05, 0.1, 0.05, 0.03, 0.02],
            [0.05, 0.2, 0.1, 0.05, 0.05, 0.15, 0.05, 0.05, 0.05, 0.1, 0.05, 0.1]
        ])

        # Criar o modelo HMM
        self.model = hmm.CategoricalHMM(n_components=self.n_states)
        self.model.startprob_ = self.state_probability
        self.model.transmat_ = self.transition_probability
        self.model.emissionprob_ = self.emission_probability

    def predict_probabilities(self, observation_sequence):
        # Prever as probabilidades de cada estado oculto para a sequência de observações
        state_probabilities = self.model.predict_proba(observation_sequence)

        # Calcular a soma das probabilidades de "Compra" (estado 0) e "Não Compra" (estado 1)
        total_prob_compra = np.sum(state_probabilities[:, 0])  # Soma das probabilidades de "Compra"
        total_prob_nao_compra = np.sum(state_probabilities[:, 1])  # Soma das probabilidades de "Não Compra"

        # Normalizar para obter a probabilidade final agregada
        total_prob = total_prob_compra + total_prob_nao_compra
        prob_compra_agregada = (total_prob_compra / total_prob) * 100
        prob_nao_compra_agregada = (total_prob_nao_compra / total_prob) * 100

        return prob_compra_agregada, prob_nao_compra_agregada
