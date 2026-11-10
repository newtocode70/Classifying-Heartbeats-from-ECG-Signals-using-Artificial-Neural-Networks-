# ECG Heartbeat Classification Using a Neural Network Built from Scratch

A Machine Learning project that classifies ECG heartbeats into five categories using a fully connected neural network implemented from scratch with PyTorch.

## Project Overview

Electrocardiography (ECG) records the electrical activity of the human heart. This project uses machine learning to automatically classify ECG heartbeat signals into five different categories.

The model receives 187 ECG signal values as input and predicts one of five heartbeat classes.

### Heartbeat Categories

| Label | Class | Description |
|---|---|---|
| N | Normal | Normal heartbeat |
| S | Supraventricular | Supraventricular ectopic heartbeat |
| V | Ventricular | Ventricular ectopic heartbeat |
| F | Fusion | Fusion heartbeat |
| Q | Unknown/Paced | Unknown or paced heartbeat |

## Dataset

**Dataset:** MIT-BIH ECG Heartbeat Categorization Dataset

The dataset contains two CSV files:

- `mitbih_train.csv` – 87,554 samples
- `mitbih_test.csv` – 21,892 samples

Each row contains 188 columns:

- First 187 columns: ECG signal features.
- Last column: Heartbeat class label.

The original training dataset was divided into:
- 90% training data.
- 10% validation data.

The separate test dataset was used for final evaluation.

**Dataset download:** https://www.kaggle.com/datasets/shayanfazeli/heartbeat

Download the two CSV files and place them inside the `data/` folder.

## Neural Network Architecture

The neural network contains three trainable layers.

**Architecture:**

`187 → 128 → 64 → 5`

| Layer | Neurons | Activation |
|---|---:|---|
| Input | 187 | None |
| Hidden Layer 1 | 128 | ReLU |
| Hidden Layer 2 | 64 | ReLU |
| Output | 5 | Logits |

Total trainable parameters: **32,645**

The weights were initialized using He initialization, and biases were initialized to zero.

The network was implemented using PyTorch without using ready-made linear layers or optimizers.

## Methodology

The project follows these steps:

1. Load the MIT-BIH ECG dataset.
2. Separate ECG signals and labels.
3. Prepare training, validation, and testing data.
4. Build the neural network architecture manually.
5. Implement forward propagation.
6. Implement cross-entropy loss using `torch.logsumexp`.
7. Verify gradients using manual calculations and PyTorch Autograd.
8. Train the model using mini-batch gradient descent with momentum.
9. Compare different learning rates.
10. Evaluate the selected model using the test dataset.
11. Develop a Streamlit frontend for heartbeat predictions.

## Training Configuration

| Parameter | Value |
|---|---:|
| Batch Size | 128 |
| Momentum | 0.9 |
| Weight Decay | 0.0001 |
| Epochs | 30 |
| Learning Rates | 0.01, 0.1, 0.5 |
| Selected Learning Rate | 0.5 |

The best model was selected using validation loss.

Best validation loss: **0.0729**

Best epoch: **23**

## Model Performance

The selected model was evaluated on 21,892 test samples.

| Evaluation Metric | Result |
|---|---:|
| Correct Predictions | 21,399 |
| Total Predictions | 21,892 |
| Test Accuracy | 97.75% |
| Balanced Accuracy | 83.76% |

### Per-Class Accuracy

| Heartbeat Class | Accuracy |
|---|---:|
| N – Normal | 99.51% |
| S – Supraventricular | 68.17% |
| V – Ventricular | 90.95% |
| F – Fusion | 62.35% |
| Q – Unknown/Paced | 97.82% |

The model achieved high overall accuracy. However, the lower accuracy of S and F classes indicates that class imbalance remains a challenge.

### Confusion Matrix

Rows represent actual classes, and columns represent predicted classes.

| Actual / Predicted | N | S | V | F | Q |
|---|---:|---:|---:|---:|---:|
| N | 18029 | 27 | 16 | 10 | 36 |
| S | 168 | 379 | 6 | 0 | 3 |
| V | 104 | 4 | 1317 | 14 | 9 |
| F | 39 | 0 | 21 | 101 | 1 |
| Q | 29 | 2 | 4 | 0 | 1573 |

## Streamlit Web Application

A Streamlit frontend was developed to make the model easy to use.

The application allows users to:

- Upload an ECG heartbeat containing 187 signal values.
- Select a sample from the test dataset.
- Visualize the ECG waveform.
- View the predicted heartbeat class.
- View class probabilities for all five categories.

### Run the Application

From the project root directory, install the required libraries:

```bash
python -m pip install -r requirements.txt
```

Run the frontend:

```bash
streamlit run app/app.py
```

The application will open in your browser.

## Installation and Execution

### 1. Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the Project Directory

```bash
cd ECG-Heartbeat-Classification
```

### 3. Install Requirements

```bash
python -m pip install -r requirements.txt
```

### 4. Download the Dataset

Download `mitbih_train.csv` and `mitbih_test.csv` from Kaggle and place both files inside the `data/` folder.

### 5. Run the Project

Run the following commands from the project root:

```bash
python -m src.load_data
python -m src.plot_beats
python -m src.prepare
python -m src.grad_check
python -m src.train
python -m src.test_model
```

Note: Running `src.train` trains the models again and generates the training results.

## Project Structure

```text
ECG-Heartbeat-Classification/
│
├── data/
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── plot_beats.py
│   ├── prepare.py
│   ├── model.py
│   ├── grad_check.py
│   ├── train.py
│   ├── metrics.py
│   └── test_model.py
│
├── results/
│   ├── analysis.md
│   ├── best_model.pth
│   └── generated figures and tables
│
├── app/
│   └── app.py
│
├── Assignment_3.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- Matplotlib
- Streamlit
- Jupyter Notebook
- Git and GitHub

## Limitations and Future Improvements

Although the model achieved 97.75% test accuracy, the lower class-wise accuracy for S and F indicates that further improvements are possible.

Future improvements include:

- Handling class imbalance.
- Improving classification of rare heartbeat categories.
- Testing on additional ECG datasets.
- Exploring advanced neural network architectures.
- Improving the frontend visualization.

## Team Members

**Project Contributors:**

1. Kashif Gulzar
2. Muhammad Umar Draz

## Academic Project

Machine Learning – Assignment 3

Fall 2026

FAST National University of Computer and Emerging Sciences

---

*This project was developed for academic and educational purposes. It is not intended to replace professional medical diagnosis.*