# PRNN: Experiment1  


## Overview  

The experiment covers both regression and classification tasks, where we applied a variety of machine learning techniques and evaluated their performance using various metrics. The tasks include performing regression with linear and kernelized methods, as well as classifying data using Bayesian classifiers, K-nearest neighbors, and linear classifiers.

---

## Results Summary  

### 1. Regression Tasks  

#### **Multilinear Regression**  
- Implemented a **linear regression model** to predict the 3D position of a particle using a 10-dimensional feature vector.  
- Metrics used for evaluation:  
  - Pearson Correlation  
  - Mean Squared Error (MSE)  
  - Mean Absolute Error (MAE)  
- Visualized the performance using correlation plots comparing the actual vs predicted values.  

#### **Generalized Regression with Polynomial Kernel**  
- Implemented a **generalized regression** using a polynomial kernel to predict the particle's position based on the magnitude of the force along two basis vectors.  
- Evaluated the model with the same metrics and presented correlation plots for each target variable.  

#### **Generalized Regression with Non-Polynomial Kernel**  
- Implemented **generalized regression** with a non-polynomial kernel to predict the probability of rain using 5 satellite readings.  
- Reported the Pearson Correlation, MSE, and MAE, along with correlation plots for the target variable.  

---

### 2. Classification Tasks  

#### **Binary Classification**  
- Implemented **binary classification** for predicting the product being produced based on the sensor readings from 10 sensors.  
- Used the following methods:  
  - Bayes' classifiers (Normal, Exponential, and GMMs)  
  - K-nearest neighbors (K-NN) with different values of K and distance metrics (Euclidean and Cosine)  
  - Linear classifiers (One-vs-Rest for multi-class cases)  
- Evaluation metrics included:  
  - Classification accuracy  
  - Confusion matrix  
  - F1 score  
  - ROC curves  

#### **Multi-Class Classification (10 Classes)**  
- Implemented **multi-class classification** for predicting one of 10 products based on sensor data from 25 sensors.  
- Used the same methods as in Q4, but extended for 10 classes.  
- Evaluated models using the following metrics:  
  - Classification accuracy  
  - Confusion matrix  
  - F1 score  
  - ROC curves for class pairs  
  - Likelihood curve for EM with different mixture counts  
  - Empirical risk using logistic regression  

---


## Key Insights  

- **Regression Tasks:**  
  - Linear regression effectively modeled the relationship between the sensor readings and the 3D position of the particle.  
  - The polynomial kernel regression improved predictions for tasks involving nonlinear relationships.  
  - The non-polynomial kernel regression successfully predicted the probability of rain, highlighting the importance of kernel choice.  

- **Classification Tasks:**  
  - The **Bayesian classifiers** and **K-NN** models showed diverse strengths depending on the dataset's characteristics.  
  - **One-vs-Rest classifiers** performed well for multi-class classification, and we observed how different kernels affected the performance of the models.  
  - The **logistic regression** model provided a solid baseline for evaluating the empirical risk on both train and test data.  

---






# PRNN: Experiment2 

---

## Overview  

The tasks include implementing backpropagation for neural networks, solving regression and classification problems, and analyzing the Kuzushiji-MNIST dataset. The work emphasizes fundamental understanding and experimentation with machine learning algorithms without relying on advanced ML libraries.  

---

## Results Summary  

### 1. General Tasks  

- Implemented **error backpropagation algorithm** for:  
  - Fully connected multi-layer feed-forward neural networks (MLPs).  
  - Multilayer Convolutional Neural Networks (CNNs).  
- Incorporated hyperparameters such as loss functions, layer configurations, and kernel properties.  
- Demonstrated bias-variance trade-offs with **three regularization techniques** and visualized results using bias-variance curves.  

### 2. Regression Tasks  

- Solved regression problems using **multi-layer perceptrons (MLPs)**.  
- Achieved results comparable to Assignment 1 metrics while demonstrating improvements through overfitting and regularization.  

### 3. Classification Tasks  

- Used MLPs for classification tasks with two distinct loss functions.  
- Implemented **SVMs** with and without slack formulations and experimented with three kernels:  
  - Linear Kernel  
  - Polynomial Kernel  
  - Gaussian Kernel  
- Conducted grid search for optimal hyperparameters and applied the **one-vs-rest approach** for multi-class classification.  

### 4. Kuzushiji-MNIST Dataset  

- Performed a **10-class classification** using:  
  - Logistic Regression.  
  - SVM with Gaussian Kernel.  
  - Multi-layer Perceptrons (MLPs).  
  - Convolutional Neural Networks (CNNs).  
- Compared models based on **size vs performance**:  
  - CNNs outperformed other methods with the highest accuracy and efficiency for the dataset.  

---





## Key Insights  

- **Backpropagation Implementation:** Developed a robust modular implementation for both MLPs and CNNs, enabling flexibility in hyperparameter adjustments.  
- **Regularization Techniques:** Demonstrated the effectiveness of regularization in mitigating overfitting through bias-variance plots.  
- **SVM Analysis:** Explored kernelized SVMs and highlighted the trade-offs between computational complexity and model performance.  
- **Kuzushiji-MNIST Performance:** Established CNNs as the most efficient model for high-dimensional image classification, showcasing their scalability and accuracy.  

---


# PRNN- Experiment3  

**Professor:** Prof. Prathosh A. P  
**Date:** April 1, 2024  

---

## Overview  

Experiment3 which involves working with two distinct datasets: one on vision (Animal Image Dataset) and one on text (News Text Dataset). The assignment tasks include implementing machine learning algorithms from scratch, including self-attention, PCA, K-means, decision trees, and gradient boosting. The assignment also requires solving classification and clustering problems for both datasets.

---

## Results Summary  

### 1. General Tasks  

#### 1.1 Self-Attention Block  
- Implemented a **self-attention block** from scratch using token length and number of attention layers as hyper-parameters.  

#### 1.2 PCA Implementation  
- Developed a **Principal Component Analysis (PCA)** class that allows for dimensionality reduction with the number of components as a hyper-parameter.  

#### 1.3 K-means Clustering  
- Implemented **K-means clustering** with different distance metrics (e.g., Euclidean, Cosine) as hyper-parameters for clustering tasks.  

#### 1.4 Decision Tree Classifier  
- Built a **Decision Tree Classifier** with multiple impurity functions (Gini, Entropy) as hyper-parameters.  

#### 1.5 Gradient Boosting  
- Implemented **gradient boosting** with the ability to take multiple classifiers as inputs and perform ensemble learning.  

---

### 2. Vision Dataset (Animal Image Dataset)  

#### 2.1 10-Class Classification with CNN  
- Applied a **Convolutional Neural Network (CNN)** to classify images from the Animal Image Dataset into 10 classes.  
- Evaluated the model using **accuracy** and **F1 score** as metrics.  

#### 2.2 PCA + MLP for Dimensionality Reduction  
- Used **PCA** to reduce the feature dimensions and applied a **Multilayer Perceptron (MLP)** on the reduced features.  
- Compared the results with the CNN model.  

#### 2.3 Transformer Model with Self-Attention  
- Implemented a **transformer model** with self-attention using the PCA-reduced features and compared its performance with the CNN and MLP.  

#### 2.4 K-means Clustering on Raw and PCA’ed Data  
- Applied **K-means clustering** on both raw image pixels and PCA-reduced feature data.  
- Compared the clustering results using **Normalized Mutual Information (NMI)** as the metric.  

#### 2.5 Ensemble Models  
- Implemented an **ensemble model** combining **CNN**, **MLP**, and **Decision Trees** in an **AdaBoost framework**.  
- Compared the ensemble model's performance with the non-ensemble models.  

#### 2.6 Feature Extraction from Pre-trained Imagenet  
- Extracted features from a pre-trained **Imagenet** model and trained an MLP on top of the extracted features.  

---

### 3. Text Dataset (News Text Data)  

#### 3.1 12-Class Classification using MLP  
- Solved a **12-class classification problem** using an **MLP** with features derived from news headlines using **TF-IDF** embeddings.  
- Padded the features with zeros to ensure uniform input size.  

#### 3.2 Transformer Model with Self-Attention for Text  
- Applied a **transformer model** with self-attention to the same **12-class classification problem** and compared results with the MLP model.  

#### 3.3 Random Forest and Gradient Boosted Trees  
- Implemented **Random Forest** and **Gradient Boosted Trees** for the classification task.  
- Compared the results with the MLP and Transformer models.  

---


## Key Insights  

- **Vision Dataset:**  
  - The **CNN** provided strong performance for image classification with high accuracy and F1 scores.  
  - **PCA + MLP** performed well in dimensionality reduction, but CNN outperformed in the raw feature space.  
  - The **Transformer model** with self-attention improved performance slightly over the MLP but was computationally more intensive.  
  - **Ensemble models** (CNN/MLP/Decision Trees in AdaBoost) demonstrated improved generalization over individual models.  
  - **Feature extraction from Imagenet** followed by MLP training provided strong performance but required careful feature handling.

- **Text Dataset:**  
  - The **MLP** model performed well with TF-IDF embeddings for 12-class classification.  
  - The **Transformer model** with self-attention achieved comparable results but was more complex to implement.  
  - **Random Forest** and **Gradient Boosting** offered good performance but were not as effective as the transformer for text classification tasks.  

---



## Author  

- Sriya  
---

## Notes  

- Ensure you have **numpy** and **matplotlib** installed in your environment.  
- LIBSVM is required for the SVM tasks.  
- For **TF-IDF embeddings**, the **Sklearn** library was used for feature extraction.  
- Adhere to academic integrity policies while using this repository.  

---
