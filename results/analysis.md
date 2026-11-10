# Task 6 – Analysis

## 6.1 Always-N baseline and the limitation of accuracy

The test dataset contains 21,892 heartbeat records. According to the class distribution, 18,118 of these records belong to class N (Normal). An Always-N classifier predicts N for every heartbeat. It is therefore correct only for the N records.

Accuracy = (18118/21892) x100
82.76% baseline accuracy.
Our neural network achieved a test accuracy of **97.75%**, which is approximately **14.99 percentage points** higher than the Always-N baseline. This shows that the network performs much better than a classifier that predicts only the majority class.

However, accuracy alone does not show how well the model recognises every heartbeat type. The dataset is imbalanced: class N has many more examples than class F or class S. A model can obtain a high overall accuracy while still making many errors on rare classes. Our results demonstrate this: the per-class accuracy is 99.51% for N, 68.17% for S, 90.95% for V, 62.35% for F, and 97.82% for Q. The balanced accuracy is **83.76%**, which is lower than the overall test accuracy of 97.75%.

Therefore, overall accuracy should be considered together with the confusion matrix, per-class accuracy, and balanced accuracy.

## 6.2 Lowest per-class accuracy and its most common confusion

Class **F (Fusion)** has the lowest per-class accuracy at **62.35%**. In the confusion matrix, the actual F row contains the following predictions:

| Actual class F predicted as | Number of records |
|---|---:|
| N | 39 |
| S | 0 |
| V | 21 |
| F | 101 |
| Q | 1 |

The most common mistake is classifying an F beat as **N (Normal)**: 39 F records were predicted as N. The next most common mistake is predicting V (Ventricular), with 21 records.

Fusion beats combine characteristics of different beat types, particularly normal and ventricular activity. Because their ECG shapes can share features with normal beats, some F examples may look similar to N. This overlap can make the classes difficult for the network to separate. The small number of F training examples (641) also gives the network fewer examples from which to learn the variation within this class.

## 6.3 Training and validation loss curves

The training and validation loss curves for the selected learning rate, **0.5**, both decrease substantially during training. The training loss continues to decrease and reaches approximately 0.05 near the final epoch. The validation loss also decreases overall, but it fluctuates more and remains above the training loss during much of the later training.

This creates a modest gap between training and validation loss. The gap suggests **some mild overfitting**: the network fits the training data better than the validation data. However, the validation loss does not increase continuously, and the validation accuracy remains high and close to the training accuracy. Therefore, the curves do not indicate severe overfitting.

Overall, the model learns useful patterns and generalises well, but the loss gap and validation fluctuations suggest that a small amount of overfitting is present.

## 6.4 Why weights are not initialised to zero and why He initialisation is used

All weights should not be initialised to zero because neurons in the same layer would start with identical weights. They would receive identical gradients and continue learning the same features. This is called the **symmetry problem**. Different initial weights allow neurons to learn different patterns from the input.

For layers that use the ReLU activation function, He initialisation uses a weight variance of:

\[
\operatorname{Var}(W)=\frac{2}{n_{\text{in}}}
\]

where \(n_{\text{in}}\) is the number of inputs to the layer. Equivalently, the standard deviation is \(\sqrt{2/n_{\text{in}}}\) when weights are sampled from a zero-mean normal distribution.

ReLU sets negative activations to zero, so roughly half of the activations may be removed. The factor of 2 helps compensate for this reduction and keeps the signal variance at a suitable scale as it passes through ReLU layers. This supports more stable forward activations and backpropagated gradients during training.

## 6.5 Effect of momentum and weight decay

**Momentum:** Momentum uses the current mini-batch gradient together with the previous velocity:

\[
v_t=\beta v_{t-1}+(1-\beta)g_t
\]

If two consecutive mini-batch gradients point in opposite directions, their effects partly cancel in the velocity. This reduces sudden changes and oscillations in the updates. Momentum therefore makes learning smoother, although the previous velocity can still influence the update.

**Weight decay:** For a weight parameter \(P\), weight decay adds a term proportional to the weight to its gradient:

\[
g_t=\frac{\partial J}{\partial P}+\lambda P
\]

If the data-loss gradient for a weight is zero, the weight-decay term \(\lambda P\) can still produce an update. It gradually pulls the weight toward zero. This discourages unnecessarily large weights and can help reduce overfitting. In this assignment, weight decay is applied to weights, not biases.

## 6.6 A possible improvement for the rare F class

One possible improvement is to use a **class-weighted cross-entropy loss**. The F class has only 641 training examples, so it contributes much less frequently to ordinary training batches than the majority N class. A class-weighted loss would assign a larger penalty to mistakes on under-represented classes, especially F.

As a result, errors on F examples would have more influence on the model's updates. This could encourage the network to learn features that distinguish Fusion beats instead of favouring the majority class. The class weights should be selected carefully and evaluated using per-class accuracy and balanced accuracy, because excessive weighting may increase false predictions for the rare class.

This is a proposed change only; it is not implemented or tested in this assignment.

---

## Overall conclusion

The neural network achieved **97.75% test accuracy** and **83.76% balanced accuracy**. It recognises N and Q very well, while S and especially F remain more challenging. The confusion matrix shows that F is most often mistaken for N. The training curves show good learning with a modest training-validation loss gap, suggesting mild overfitting rather than a major generalisation problem. Future work could focus on improving minority-class recognition through class-weighted loss and evaluating the result with balanced accuracy and per-class metrics.
