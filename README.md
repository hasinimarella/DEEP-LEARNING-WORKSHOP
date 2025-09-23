# DEEP-LEARNING-WORKSHOP

# AIM:
To build a PyTorch-based neural network that predicts whether an individual earns more than $50,000 annually using demographic and occupational data from the Census Income dataset.

# PROCEDURE:
## 1.Data Loading and Exploration
The project begins with loading the Census Income dataset (income.csv) into a Pandas DataFrame. The dataset contains 30,000 entries, each representing demographic and occupational attributes. Initial exploration involves viewing the first and last few rows, checking the total number of entries, and analyzing the distribution of the target label (label). This step ensures that the dataset is clean, structured, and suitable for further processing.
## 2.Feature Identification and Categorization
The dataset is divided into three types of features. Categorical features include sex, education, marital-status, workclass, and occupation, which will later be processed using embedding layers. Continuous features, such as age and hours-per-week, will be normalized and used directly in the model. The target feature, label, represents the binary income class. Proper categorization ensures that each type of data is handled appropriately in the model.

## 3.Data Preprocessing
Categorical features are converted into numerical codes to make them compatible with the embedding layers in PyTorch. The dataset is shuffled to remove any ordering bias and the index is reset for consistency. Next, categorical and continuous values are stacked into tensors, with categorical tensors of type long and continuous tensors of type float32. Labels are also converted into long tensors to facilitate classification. This preprocessing step prepares the data for model input.

## 4.Train-Test Split
The dataset is split into training and testing subsets to evaluate model performance. The training set contains 25,000 entries, which are used for learning model parameters. The testing set contains 5,000 entries, reserved for evaluating how well the trained model generalizes to unseen data. This separation ensures that the model is not evaluated on the data it has already seen, providing a true measure of its predictive ability.

## 5.Model Design – TabularModel
A custom PyTorch neural network, TabularModel, is implemented to handle tabular data. The model uses embedding layers for each categorical feature to learn dense vector representations. Continuous features are batch-normalized to stabilize training. Both embeddings and continuous features are concatenated into a single vector and passed through fully connected layers with ReLU activation. Dropout layers are applied to reduce overfitting, and the final output layer has two neurons for binary classification. This architecture allows the model to effectively capture relationships between categorical and continuous features.

## 6.Model Compilation
Before training, the model is compiled by defining the loss function and optimizer. CrossEntropyLoss is used to measure the difference between predicted and actual labels. The Adam optimizer, with a learning rate of 0.001, is used to adjust model weights efficiently. A random seed is set to ensure reproducibility of results across multiple runs.

## 7.Model Training
The model is trained for 300 epochs. In each epoch, the model performs a forward pass to predict outcomes for the training data, computes the loss, backpropagates the error to calculate gradients, and updates the model parameters using the optimizer. Training loss is recorded at each epoch to monitor progress, and intermediate loss values are printed periodically. Over time, a decreasing loss indicates that the model is learning and improving its predictions.

## 8.Training Loss Visualization
After training, a loss curve is plotted using Matplotlib to visualize how the training loss decreases over epochs. The plot provides a clear representation of model convergence and helps identify potential issues such as underfitting or overfitting. This visual feedback is essential for understanding the training dynamics.

## 9.Model Evaluation
The trained model is evaluated on the test set to measure its performance on unseen data. The model is switched to evaluation mode to disable dropout and other training-specific behaviors. Predictions are generated for the test data, and CrossEntropyLoss is calculated to quantify prediction error. Accuracy is computed by comparing predicted labels with actual labels, providing a clear metric of model performance. Example results might show that the model correctly predicts approximately 80% of the test samples.


## RESULT:
Thus the binary classification model using PyTorch to predict whether an individual earns more than $50,000 annually has been successfully executed.

