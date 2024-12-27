# Modelo de Probabilidade de Compra com HMM

Este projeto implementa um modelo de Máquina de Estados Ocultos (HMM) para prever a probabilidade de um usuário realizar uma compra com base em uma sequência de observações. O modelo é capaz de analisar informações sobre o comportamento de compra e fornecer uma probabilidade de "Compra" ou "Não Compra" com base em um conjunto de variáveis, como necessidade, dinheiro disponível, satisfação e outros fatores.

## Tecnologias Utilizadas

- **Python 3.x**
- **hmmlearn** - Biblioteca para implementação de HMM
- **NumPy** - Biblioteca para manipulação de arrays
- **Tkinter** - Biblioteca para criação da interface gráfica

## Funcionalidade

A aplicação consiste em duas partes principais:
1. **Modelo HMM**: Um modelo de Máquina de Estados Ocultos que utiliza um conjunto de probabilidades de transição e emissão para prever os estados ocultos (Compra ou Não Compra) com base nas observações fornecidas.
2. **Interface Gráfica (UI)**: A interface gráfica permite que o usuário selecione as variáveis que descrevem sua situação de compra e, com base nessas observações, o modelo HMM calcula a probabilidade de realizar uma compra.

### Observações Possíveis:
- Necessidade Alta
- Necessidade Baixa
- Pouco Dinheiro
- Muito Dinheiro
- Satisfação Alta
- Satisfação Baixa
- Com Limite
- Sem Limite
- Poucos Gastos
- Muitos Gastos
- Última Compra Correta
- Última Compra Errada

### Estados Ocultos:
- Compra
- Não Compra

## Como Usar

### Pré-requisitos

Antes de rodar o projeto, você precisa ter o Python 3.x instalado no seu sistema. Além disso, é necessário instalar algumas dependências.

### Instalação

1. Clone o repositório:

    ```bash
    git clone https://github.com/SeuUsuario/probabilidade-compra-hmm.git
    cd probabilidade-compra-hmm
    ```

2. Crie e ative um ambiente virtual (opcional, mas recomendado):

    ```bash
    python -m venv venv
    source venv/bin/activate  # Para Linux/Mac
    venv\Scripts\activate     # Para Windows
    ```

3. Instale as dependências:

    ```bash
    pip install -r requirements.txt
    ```

    Ou instale as dependências manualmente:

    ```bash
    pip install numpy hmmlearn matplotlib seaborn
    ```

### Executando o Projeto

1. Após instalar as dependências, basta rodar o script Python principal:

    ```bash
    python app.py
    ```

2. A interface gráfica será aberta. Nela, você poderá selecionar as observações que descrevem sua situação (por exemplo, "Necessidade Alta", "Muito Dinheiro", "Com Limite") e clicar em "Calcular Probabilidade" para obter a probabilidade de compra.

## Estrutura do Projeto

```plaintext
probabilidade-compra-hmm/
├── main.py           # Arquivo principal para rodar o projeto
├── requirements.txt  # Arquivo com as dependências do projeto
└── README.md         # Este arquivo
