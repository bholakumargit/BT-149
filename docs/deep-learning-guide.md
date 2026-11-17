# Deep Learning for Malware Classification

## A beginner's guide to this project

This guide explains what the project trains, how each neural-network concept
works, why the data is split in a particular way, and how to read the results.
It describes the code in `scripts/train_models.py`; it does not claim that the
model is a production antivirus product.

## 1. What the project predicts

CIC-MalMem-2022 is a table of numeric memory-forensics features. A feature is
a measured property, such as a process count, handle count, or memory-module
statistic. The CSV has 58,596 rows: 55 numeric features, a `Class` label, a
`Category` label, and a `Filename` identifier.

The training script fits two separate models:

- **Detection model:** predicts `Class`, either `Benign` or `Malware`.
- **Family model:** considers only malware rows and predicts one of 15 family
  labels in `Category`, such as `Ransomware-Ako` or `Trojan-Zeus`.

The family model does not predict `Benign`; detection is its separate first
step. This separation makes each model's target clear. An application that
combines their predictions must first run detection, and should only run
family classification when malware is detected.

The project uses extracted features only. It does not execute malware files.

## 2. The input columns and labels

`Class` and `Category` are answers used during training, not input features.
`Filename` is an identifier, not a measurement. Including any of these columns
as a model input would let the model cheat by learning the answer or a sample
name instead of learning behavior from the 55 measurements.

The script therefore builds the feature list by excluding all three columns.
It converts each feature to a number and stops with an error if values are
missing, non-numeric, or infinite. It does not silently guess how to repair
bad input data.

The category labels include `Benign` plus 15 malware families. The
ransomware/spyware/trojan group can be read from a malware category's prefix,
but the trained family model predicts the full family name.

## 3. Train, validation, and test data

Evaluating a model on the same examples it learned from gives an overly
optimistic score. The script separates examples into three sets:

- **Training set (about 80%):** updates the model's weights.
- **Validation set (about 10%):** helps choose when to stop training.
- **Test set (about 10%):** measures the final model on examples not used to
  update or select it.

The proportions are approximate because related samples must stay together.
The split is stratified, meaning it tries to preserve each label's proportion
in each set, and grouped, meaning every group goes to just one set.

### Why grouping matters here

The malware filenames contain a sample hash followed by a numbered memory
dump suffix such as `-1.raw` or `-10.raw`. The importer groups rows by the
common filename after removing this suffix. Thus the ten dumps for one sample
cannot be divided across training and test data. Otherwise, the model could
recognize a near-duplicate sample in the test set and appear more accurate
than it is on a genuinely new sample.

In this copy of the dataset, all benign rows have `Filename` equal to the
placeholder `Benign`; it does not identify individual benign samples. The
script therefore treats each benign row as its own group so it can put benign
examples into all three sets. This is a limitation: the data does not provide
enough benign identifiers to guarantee that related benign captures are
separated. Better per-capture IDs would strengthen that evaluation.

## 4. What an MLP is

An **MLP** (multilayer perceptron) is a neural network made from fully
connected layers. "Fully connected" means each neuron in a layer receives
values from every neuron in the previous layer. It is a sensible baseline for
a fixed-length numeric feature vector.

This project uses the following sequence:

1. **Input:** 55 standardized feature values.
2. **Dense layer:** 128 neurons.
3. **ReLU:** a nonlinear activation function.
4. **Dropout:** randomly masks 25% of activations during training.
5. **Dense layer:** 64 neurons.
6. **ReLU and dropout:** another nonlinear transformation and regularization.
7. **Output layer:** one score per class (2 for detection, 15 for family).

The dense layers learn combinations of features. The first layers can learn
useful intermediate patterns; the output layer combines those patterns into
class scores. An MLP does not automatically understand file contents,
sequences, or images: it only sees the numeric columns supplied to it.

### ReLU activation

ReLU means `max(0, x)`: negative inputs become zero, positive inputs pass
through. Without a nonlinear activation, stacking dense layers would still
behave like one linear transformation and would have limited ability to learn
complex feature relationships.

### Logits and softmax

The output layer returns **logits**, which are unnormalized class scores.
Cross-entropy loss uses those scores directly during training. For reporting
probabilities, softmax converts them to values between 0 and 1 that sum to 1.
The predicted class is the output with the largest probability.

## 5. Scaling the numeric features

Feature columns can have very different numeric ranges. For example, a count
may be in the thousands while an average is a decimal close to zero.
`StandardScaler` transforms each feature using:

`scaled value = (value - training mean) / training standard deviation`

The mean and standard deviation are learned using the training set only. The
same saved values are then applied to validation and test rows. Learning
scaling statistics from all rows would leak information from the evaluation
sets into training.

The learned means and scales, as well as the feature names and their order,
are saved with each model. A future prediction implementation must use the
same feature order and scaling values.

## 6. Learning the weights

Training starts with randomly initialized weights. For a batch of examples:

1. **Forward pass:** the features move through the dense layers to produce
   logits.
2. **Loss:** cross-entropy compares logits with the correct labels. A lower
   loss means the model assigned more probability to the correct answers.
3. **Backpropagation:** gradients estimate how each weight contributed to
   the loss.
4. **Optimizer step:** AdamW uses those gradients to adjust the weights.

The script uses **AdamW**, a commonly used optimizer that adapts update sizes
for individual weights and applies weight decay. The learning rate is `0.001`;
it controls the size of weight updates. Weight decay is `0.0001` and gently
discourages unnecessarily large weights.

The batch size is 256. A batch is the number of training rows used to compute
one optimizer update. The training set is shuffled each epoch so batches do
not always contain the same ordering of examples.

### Class weights

The malware-family labels are not exactly equally frequent. Cross-entropy is
therefore weighted inversely to each class's training frequency. This reduces
the chance that the model can get a good average loss by favoring common
families and ignoring less common ones. Weights are calculated from training
labels only.

## 7. Overfitting, dropout, and early stopping

**Overfitting** happens when a model learns details specific to training rows
that do not generalize to new rows. Training loss can keep improving even
while validation performance gets worse.

**Dropout** is active during training: it randomly sets some layer outputs to
zero, encouraging the network not to rely on a single narrow path. During
evaluation and prediction, dropout is switched off.

**Early stopping** monitors validation loss. The script remembers the weights
with the best validation loss and stops if that score fails to improve for
eight epochs. It tries at most 50 epochs. The final saved weights are restored
from the best validation epoch, not necessarily the last epoch.

An epoch is one pass over the complete training set. Training loss and
validation loss are printed for each epoch so the learning process is visible.

## 8. Metrics and what they mean

The test set is used after model selection to calculate:

- **Accuracy:** fraction of test examples predicted correctly.
- **Precision:** among predictions for a class, how many were correct.
- **Recall:** among actual examples of a class, how many were found.
- **F1-score:** harmonic mean of precision and recall.
- **Macro F1:** calculate F1 per class, then average classes equally. This
  makes performance on rare classes visible.
- **Weighted F1:** average F1 weighted by how many examples each class has.
- **Confusion matrix:** counts actual labels against predicted labels.
- **ROC-AUC (detection only):** measures how well malware scores rank malware
  above benign examples over possible thresholds; 1 is ideal and 0.5 is
  roughly chance ranking.

No single metric tells the whole story. Read per-class precision, recall, and
the confusion matrix, especially for the malware families that matter to the
research question. A high accuracy alone is not proof of a useful detector.

## 9. Run training and find its outputs

From the project folder in PowerShell:

```powershell
python -m pip install -r requirements-training.txt
python scripts/train_models.py
```

Training automatically uses CUDA if PyTorch can access a compatible GPU;
otherwise it runs on CPU. Each model is saved under `models/`:

- `detection_model.pt` - binary detector checkpoint.
- `classification_model.pt` - 15-family classifier checkpoint.
- `training_metrics.json` - split sizes, label counts, per-class scores, and
  configuration for both workflows.

The checkpoint contains the neural-network weights, architecture details,
class names, feature order, normalization values, split sizes, and test
metrics. The fixed seed (42) makes splits and initialization repeatable, but
different hardware or library versions can still produce small numerical
differences.

The dashboard reports when both models are trained and accepts one CSV row
with exactly the trained numeric feature columns. It runs detection first and
only predicts a malware family if the detection model predicts malware. A
checkpoint without its matching feature columns cannot be used for prediction.
The probability score is the model's softmax output and should not be treated
as a calibrated probability of real-world safety.

For `.exe`, `.doc`, `.pdf`, `.apk`, and `.msi`, the separate static inspection
path reports file hashes and format-specific structural metadata without
executing or saving the upload. It accepts at most 25 MB; APK archives are
inspected under entry-count, expansion-size, and compression-ratio limits.
This feature extraction is **not** a malware verdict and does not use the
CIC-MalMem-2022 models. The `.doc` and `.msi` inspection currently confirms
their shared OLE container signature, which cannot alone distinguish those
formats. Prediction for these files would require suitable labeled datasets,
format-specific features, and separately trained and evaluated models.

## 10. Limitations and responsible use

- This is a research baseline, not a production security product or a
  guarantee that a file is safe.
- The dataset has label and collection limitations. Results on it may not
  transfer to new malware, operating systems, or memory-capture tools.
- The benign `Filename` placeholder prevents grouping benign captures by
  their real source. This weakens leakage controls for that class.
- The split groups malware dump numbers by their shared sample hash. If the
  source dataset has other related samples without a common identifier, that
  relationship cannot be detected from this CSV alone.
- Do not execute malware samples. Use only the supplied feature table and
  follow your institution's data-handling and licensing requirements.

## Glossary

- **Feature:** a measured numeric input used to make a prediction.
- **Label/target:** the correct answer the model is taught to predict.
- **Neuron:** a small calculation that combines inputs with learned weights.
- **Weight:** a number adjusted during training to improve predictions.
- **Gradient:** direction and size of a weight change suggested by the loss.
- **Batch:** a subset of training rows used for one weight update.
- **Epoch:** one complete pass through all training rows.
- **Hyperparameter:** a setting selected by the developer, such as layer
  sizes, learning rate, dropout, or batch size.
- **Checkpoint:** a saved file containing model weights and required metadata.
- **Data leakage:** information from evaluation data improperly influences
  training, making results look better than real-world performance.
