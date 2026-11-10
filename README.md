# ECG Heartbeat Classification using an Artificial Neural Network

This project develops a machine learning pipeline for classifying ECG heartbeat signals into five classes using a neural network built from scratch with PyTorch. The system processes a 187-sample heartbeat waveform and predicts the corresponding class: Normal (N), Supraventricular (S), Ventricular (V), Fusion (F), or Unknown/Paced (Q).

The repository includes the data loading pipeline, preprocessing, custom network architecture, training loop, model evaluation, result visualisation, and a Streamlit app for interactive prediction.

## Project Overview

ECG signals represent the electrical activity of the heart. Automatic heartbeat classification helps with rapid screening, anomaly detection, and clinical support. In this project, each sample is a single heartbeat represented by 187 time-domain signal values, and the network must output one of five labels.

### Target Classes

| Label | Description |
| --- | --- |
| N | Normal heartbeat |
| S | Supraventricular ectopic heartbeat |
| V | Ventricular ectopic heartbeat |
| F | Fusion heartbeat |
| Q | Unknown or paced heartbeat |

## Dataset

This project uses the MIT-BIH ECG Heartbeat Categorization dataset, which is publicly available from Kaggle.

### Files

The dataset is placed in the `data/` directory:

- `data/mitbih_train.csv` — training samples
- `data/mitbih_test.csv` — hold-out test samples

### Data Shape

Each row has 188 columns:

- Columns 1–187: ECG signal values for a single heartbeat
- Column 188: class label

### Dataset Statistics

- Training set: 87,554 records
- Test set: 21,892 records
- Validation split: 10% of the training data
- Classes: 5

The dataset is imbalanced, with the Normal class (N) being the dominant class. This creates a challenge because simple models may achieve strong overall accuracy while still struggling to detect underrepresented classes such as F and S.

## Model Architecture

The model is implemented manually with PyTorch, not using a higher-level `nn.Linear` convenience layer for the main network logic. It uses a compact feed-forward network:

`187 -> 128 -> 64 -> 5`

| Layer | Size | Activation |
| --- | ---: | --- |
| Input | 187 | None |
| Hidden 1 | 128 | ReLU |
| Hidden 2 | 64 | ReLU |
| Output | 5 | Logits |

The architecture is defined in `src/model.py` and uses He initialization for weights to work well with ReLU activations.

### Training Setup

| Parameter | Value |
| --- | ---: |
| Batch size | 128 |
| Momentum | 0.9 |
| Weight decay | 0.0001 |
| Epochs | 30 |
| Learning rates tested | 0.01, 0.1, 0.5 |
| Best learning rate | 0.5 |

The model is trained with mini-batch gradient descent and momentum, and validation loss is used to select the best-performing run.

## Methodology

The workflow is:

1. Load the raw ECG CSV files.
2. Split the training data into training and validation sets.
3. Prepare tensors for model input and labels.
4. Check gradient calculations with a manual gradient validation step.
5. Train the network using the selected learning rate.
6. Save the best model checkpoint.
7. Evaluate on the unseen test set.
8. Export metrics, confusion matrix, and plots.
9. Run a Streamlit interface for single-heartbeat prediction.

## Code Structure

```text
Assignment_3/
├── app/
│   └── app.py                  # Streamlit web app for heartbeat prediction
├── data/
│   ├── links.txt               # reference/project links
│   ├── readme.txt              # project summary and dataset notes
│   ├── mitbih_train.csv        # training dataset (download required)
│   └── mitbih_test.csv         # test dataset (download required)
├── results/
│   ├── analysis.md             # discussion and interpretation of results
│   ├── best_model.pth          # saved trained model weights
│   ├── confusion_matrix.csv   # confusion matrix data
│   ├── confusion_matrix.png   # confusion matrix plot
│   ├── example_beats.png       # representative beats from each class
│   ├── learning_rate_comparison.png
│   ├── learning_rate_results.csv
│   ├── per_class_accuracy.csv
│   ├── test_summary.csv
│   ├── training_validation_accuracy.png
│   └── training_validation_loss.png
├── src/
│   ├── __init__.py
│   ├── grad_check.py           # gradient sanity check
│   ├── load_data.py            # loads raw CSV inputs
│   ├── metrics.py              # metrics and confusion matrix utilities
│   ├── model.py                # custom neural network definition
│   ├── plot_beats.py           # plots example ECG heartbeats
│   ├── prepare.py              # train/validation/test preparation
│   ├── test_model.py            # final model evaluation and artifact export
│   └── train.py                # model training loop and result generation
├── Assignment_3.ipynb          # notebook version of the project workflow
├── ECG Classification_ Neural Network Insights _ Poster.png
├── README.md                  # project documentation
└── results/
```

## Key Code Modules

### `src/load_data.py`
Loads the training and test CSV files from `data/` and returns them as pandas DataFrames.

### `src/prepare.py`
Performs the required split and converts the raw ECG data into PyTorch tensors for model training and testing.

### `src/model.py`
Defines `ECGNet`, a fully connected neural network with custom weight initialization and a manual softmax cross-entropy implementation.

### `src/grad_check.py`
Checks the correctness of the manual gradient computation by comparing it against a known analytical result.

### `src/train.py`
Contains the training loop, validation logic, learning-rate comparison, and model checkpoint saving.

### `src/test_model.py`
Loads the best saved model, evaluates it on the test set, and saves the confusion matrix and summary metrics.

### `src/metrics.py`
Computes overall accuracy, per-class accuracy, and balanced accuracy.

### `src/plot_beats.py`
Generates example heartbeat plots, one from each class, for inspection and presentation.

### `app/app.py`
Provides a user-facing Streamlit interface where an ECG heartbeat can be uploaded or sampled from the test set and classified with a probability breakdown.

## Results

### Best Validation Performance

The learning-rate comparison produced the following results:

| Learning rate | Lowest validation loss | Best epoch | Validation accuracy at best epoch |
| --- | ---: | ---: | ---: |
| 0.01 | 0.1618 | 30 | 95.77% |
| 0.1 | 0.0857 | 28 | 97.59% |
| 0.5 | 0.0729 | 23 | 97.78% |

The learning rate of `0.5` was selected as the best-performing configuration.

### Test Set Evaluation

The selected model was evaluated on 21,892 test samples.

| Metric | Value |
| --- | ---: |
| Correct predictions | 21,399 |
| Total predictions | 21,892 |
| Test accuracy | 97.75% |
| Balanced accuracy | 83.76% |

### Comparison with an Always-N Baseline

The dataset is heavily imbalanced. The majority class N accounts for a large share of the dataset, so a naive classifier that predicts N for every sample would have an accuracy of about 82.76%.

The custom neural network reached 97.75%, which is approximately 15 percentage points higher than the Always-N baseline.

This confirms that the model is learning useful discriminative patterns beyond the majority-class behaviour.

### Per-Class Accuracy

| Class | Accuracy |
| --- | ---: |
| N – Normal | 99.51% |
| S – Supraventricular | 68.17% |
| V – Ventricular | 90.95% |
| F – Fusion | 62.35% |
| Q – Unknown/Paced | 97.82% |

As expected, the model performs extremely well on the dominant classes and weaker on rarer classes, especially `F` and `S`. This is a direct consequence of class imbalance and highlights the need for more targeted techniques in future work.

### Confusion Matrix Summary

Rows represent actual classes and columns represent predicted classes.

| Actual / Predicted | N | S | V | F | Q |
| --- | ---: | ---: | ---: | ---: | ---: |
| N | 18029 | 27 | 16 | 10 | 36 |
| S | 168 | 379 | 6 | 0 | 3 |
| V | 104 | 4 | 1317 | 14 | 9 |
| F | 39 | 0 | 21 | 101 | 1 |
| Q | 29 | 2 | 4 | 0 | 1573 |

The most difficult class is `F (Fusion)`, which is often confused with `N (Normal)`. This indicates that some fusion heartbeats share waveform characteristics with normal beats and are therefore harder to separate.

## Plot and Result Artifacts

The following outputs were generated and saved in `results/`:

- `learning_rate_comparison.png` — validation loss curves for learning rates 0.01, 0.1, and 0.5
- `training_validation_loss.png` — selected model training and validation loss curves
- `training_validation_accuracy.png` — training and validation accuracy over epochs
- `confusion_matrix.png` — confusion matrix plot
- `example_beats.png` — one waveform example per heartbeat class
- `learning_rate_results.csv` — comparison table of training runs
- `confusion_matrix.csv` — matrix values in CSV form
- `per_class_accuracy.csv` — per-class accuracy results
- `test_summary.csv` — overall evaluation summary
- `best_model.pth` — saved trained weights
- `analysis.md` — deeper interpretation and discussion of the model's behaviour

## Running the Project

### 1. Install dependencies

This repository uses Python with PyTorch, Pandas, NumPy, Matplotlib, and Streamlit.

```bash
python -m pip install numpy pandas matplotlib torch streamlit
```

### 2. Download the dataset

Download the MIT-BIH CSV files from Kaggle and place them into the `data` folder:

```text
data/
├── mitbih_train.csv
└── mitbih_test.csv
```

### 3. Train the model

```bash
python -m src.train
```

This trains the network for multiple learning rates and saves the best model to `results/best_model.pth`.

### 4. Validate the model

```bash
python -m src.test_model
```

This evaluates the saved model on the test set and writes the results to `results/`.

### 5. Plot example beats

```bash
python -m src.plot_beats
```

### 6. Run the Streamlit app

```bash
streamlit run app/app.py
```

The app lets a user upload an ECG heartbeat or select a row from the test dataset and inspect the predicted class and output probabilities.

## Key Takeaways

- The model achieves strong overall classification performance on ECG heartbeats.
- The architecture is compact, interpretable, and built from scratch with PyTorch.
- Class imbalance affects minority-class recognition, especially for `F` and `S` beats.
- The network learns meaningful patterns beyond majority-class prediction, as shown by the improvement over the Always-N baseline.
- The implementation is suitable as a baseline model and for further improvement through techniques such as class-weighted loss or augmentation.

## Limitations and Future Work

The key challenge in this project is the imbalance among heartbeat classes. Although overall accuracy is high, minority-class performance is much lower. Future work could include:

- class-weighted loss to emphasize underrepresented classes
- resampling or augmentation techniques
- deeper or more specialised architectures
- evaluation using balanced accuracy and recall-focused metrics
- model calibration and uncertainty estimation

## Conclusion

This project demonstrates that a manually implemented neural network can effectively classify ECG heartbeats with high accuracy. It provides a clear end-to-end workflow for preprocessing, training, evaluation, and deployment in a lightweight web app. The results show that strong overall performance is possible, while also highlighting the need for additional work on class imbalance and rare heartbeat classes.

## References

- MIT-BIH ECG Heartbeat Categorization dataset on Kaggle
- Project notes and analysis in `results/analysis.md`
- Repository-related links are also listed in `data/links.txt`
