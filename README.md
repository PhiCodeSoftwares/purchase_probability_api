# Purchase Probability Model with HMM

This project implements a Hidden Markov Model (HMM) to predict the probability of a user making a purchase based on a sequence of observations. The model analyzes purchase behavior data and provides probabilities for "Purchase" or "No Purchase" based on variables such as need, available money, satisfaction, and other factors.

## Technologies Used

- **Python 3.x**
- **hmmlearn** - Library for implementing HMM
- **NumPy** - Library for array manipulation
- **Flask** - Framework for the API

## Functionality

The application consists of two main parts:
1. **HMM Model**: A Hidden Markov Model that uses transition and emission probabilities to predict hidden states (Purchase or No Purchase) based on the given observations.
2. **Flask API**: An API to expose the model functionality for external access, allowing users to send observation sequences and receive predictions.

### Possible Observations:
- High Need
- Low Need
- Low Money
- High Money
- High Satisfaction
- Low Satisfaction
- With Limit
- No Limit
- Low Expenses
- High Expenses
- Correct Last Purchase
- Wrong Last Purchase

### Hidden States:
- Purchase
- No Purchase

## How to Use

### Prerequisites

Before running the project, make sure you have Python 3.x installed on your system. Additionally, you need to install some dependencies.

### Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/YourUsername/purchase-probability-hmm.git
    cd purchase-probability-hmm
    ```

2. Create and activate a virtual environment (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # For Linux/Mac
    venv\Scripts\activate     # For Windows
    ```

3. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    Or install the dependencies manually:

    ```bash
    pip install numpy hmmlearn matplotlib seaborn flask
    ```

### Running the Project

1. After installing the dependencies, start the Flask API:

    ```bash
    python api.py
    ```

2. Use any API testing tool (like Postman) or a web browser to interact with the endpoints.

    - **Get observation options:**
      ```
      GET /api/options
      ```

    - **Get purchase probabilities:**
      ```
      POST /api/purchase_accuracy
      Body (JSON):
      {
          "observation_sequence": [0, 3, 5]
      }
      ```

## Project Structure

```plaintext
purchase-probability/
├── api.py            # Flask API
├── hmm.py            # HMM Model
├── hmm_model.pkl     # Saved HMM model
├── requirements.txt  # Project dependencies
├── README.md         # This file
└── .gitignore        # Git ignore rules
```