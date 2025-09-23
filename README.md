# DEEP-LEARNING-WORKSHOP

# AIM:
To build a PyTorch-based neural network that predicts whether an individual earns more than $50,000 annually using demographic and occupational data from the Census Income dataset.

# PROCEDURE:
## 1. Data Loading and Exploration:

* The project begins by loading the Census Income dataset (income.csv) into a Pandas DataFrame.

* The dataset contains 30,000 entries, each with demographic and work-related attributes.

* Initial exploration includes:

   * Displaying the first and last few rows to understand the structure.

   * Checking the number of entries.

   * Observing the distribution of the target label (label) to identify any class imbalance.

* This step ensures that the dataset is clean and suitable for model training.

## 2. Feature Identification and Categorization:

* Features are divided into three types:

   1. Categorical features: sex, education, marital-status, workclass, occupation 
      * These features will be handled via embedding layers in the neural network.

  2. Continuous features: age, hours-per-week.
      * These will be normalized and used directly in the network.

   3. Target label: label (binary income class).

      * Categorizing features correctly allows the model to process them efficiently.

## 3. Data Preprocessing:

* Categorical Encoding: Each categorical column is converted into numerical codes to be compatible with embedding layers.

* Shuffling: The dataset is shuffled to remove any ordering bias and to ensure randomness in training and testing data.

* Index Reset: After shuffling, indices are reset for consistency.

* Tensor Conversion:

   * Categorical codes → long tensors for embeddings.

   * Continuous values → float32 tensors for batch normalization.

   * Labels → long tensors for loss computation.

## 4. Train-Test Split:

* The dataset is divided into training and testing sets:

    * Training set: 25,000 entries.

    * Testing set: 5,000 entries.

* Training data is used for learning model parameters, while testing data is reserved for evaluating model generalization.

## 5. Model Design – TabularModel

* A custom PyTorch neural network (TabularModel) is implemented to handle tabular data.

* Key components:

1. Embedding layers for categorical variables.

   * Learn dense vector representations for each category.

2. Batch Normalization for continuous features.

   * Stabilizes training by normalizing input distributions.

3. Fully connected layers with ReLU activation.

   * Process combined feature vectors to learn complex relationships.

4. Dropout layers (p=0.4) to prevent overfitting.

5. Output layer with 2 neurons for binary classification.

   * This architecture allows the model to effectively combine categorical and continuous information.

## 6. Model Compilation

* Loss Function: CrossEntropyLoss to measure prediction error for classification.

* Optimizer: Adam with learning rate 0.001 for efficient parameter updates.

* Random Seed: Set to ensure reproducibility of results.

## 7. Model Training

* The training loop runs for 300 epochs.

* Each epoch involves:

  1. Forward pass: the model generates predictions for the training data.

  2. Loss computation: difference between predictions and true labels is calculated.

  3. Backpropagation: gradients are computed.

   4. Weight updates: optimizer adjusts model parameters to reduce loss.

* Training losses are recorded after each epoch to monitor model learning.

* Intermediate losses are printed every 25 epochs for progress tracking.

## 8. Training Loss Visualization

* A loss curve is plotted using Matplotlib to visualize how the loss decreases over epochs.

* The curve provides insight into model convergence and helps identify issues such as overfitting or underfitting.

## 9. Model Evaluation

* The trained model is switched to evaluation mode (disabling dropout and other training-specific behaviors).

* Test data is passed through the model to generate predictions.

* Evaluation metrics computed:

  * Cross-Entropy Loss on test data.

   * Accuracy: percentage of correct predictions compared to actual labels.

* Example outcome:
```
CE Loss: 0.39
4012 out of 5000 = 80.24% correct
```

## RESULT:
Thus the binary classification model using PyTorch to predict whether an individual earns more than $50,000 annually has been successfully executed.

