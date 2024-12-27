import tkinter as tk
from tkinter import messagebox
from hmm import HMMModel
import numpy as np

class HMMApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Probabilidade de Compra")
        
        self.model = HMMModel()

        # Criar os widgets de UI
        self.create_widgets()

        self.root.mainloop()

    def create_widgets(self):
        # Instruções
        label = tk.Label(self.root, text="Escolha os estados de observação:")
        label.grid(row=0, column=0, columnspan=2)

        # Labels para observações
        self.obs_vars = {}
        for i, obs in enumerate(self.model.observations):
            var = tk.IntVar()
            checkbox = tk.Checkbutton(self.root, text=obs, variable=var)
            checkbox.grid(row=i+1, column=0, sticky="w")
            self.obs_vars[obs] = var

        # Botão para calcular a probabilidade
        calc_button = tk.Button(self.root, text="Calcular Probabilidade", command=self.calculate_probability)
        calc_button.grid(row=len(self.model.observations)+1, column=0, columnspan=2)

        # Label para exibir a probabilidade
        self.result_label = tk.Label(self.root, text="", font=("Arial", 14))
        self.result_label.grid(row=len(self.model.observations)+2, column=0, columnspan=2)

    def calculate_probability(self):
        # Obter a sequência de observações baseada nas seleções
        observation_sequence = []
        for obs, var in self.obs_vars.items():
            if var.get() == 1:  # Se o checkbox estiver marcado
                observation_sequence.append(self.model.observations.index(obs))

        if not observation_sequence:
            messagebox.showerror("Erro", "Por favor, selecione pelo menos uma observação.")
            return

        # Converter a sequência de observações para um formato compatível com o HMM
        observation_sequence = np.array(observation_sequence).reshape(-1, 1)

        # Calcular as probabilidades agregadas
        prob_compra, prob_nao_compra = self.model.predict_probabilities(observation_sequence)

        # Exibir o resultado
        self.result_label.config(text=f"Probabilidade de Compra: {prob_compra:.2f}%\nProbabilidade de Não Compra: {prob_nao_compra:.2f}%")
    
