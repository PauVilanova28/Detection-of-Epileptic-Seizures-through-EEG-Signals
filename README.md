# Detection of Epileptic Seizures through EEG Signals
# Introduction

This project addresses the detection of epileptic activity through the analysis of electroencephalographic (EEG) signals. EEG signals are critical for diagnosing neurological disorders like epilepsy due to their high temporal resolution. However, manual analysis of these signals is time-consuming and complex, necessitating automated techniques to improve diagnostic efficiency and precision.

## Objectives

1. **Detecting Epileptic Seizures**:
   - **Channel Fusion**: Create a spatial representation of EEG windows by fusing information from all channels. This allows classification of windows as positive (seizure) or negative (non-seizure).
   - **Contribution**: Identify spatial regions critical for seizure detection.

2. **Incorporating Temporal Information**:
   - **LSTM Networks**: Represent EEG windows in sequences of length *K*, capturing temporal relationships for classification.
   - **Contribution**: Add temporal context to enhance detection accuracy and sensitivity.

3. **Model Comparison**:
   - Evaluate the performance of the individual models (Channel Fusion and LSTM) and a combined model that integrates both approaches.

# Methodology

The project trains and evaluates three models: Channel Fusion, LSTM, and a combined model integrating both. These models are assessed through two evaluation approaches: 

1. **Population Approach**: Testing on unseen patients.
2. **Personalized Approach**: Testing on new events from the same patients used in training.

## Cross-Validation

- **Population Approach**: 24 folds, each testing a unique patient while training on the rest.
- **Personalized Approach**: 10 folds, segmenting data from the same patients into training and testing.

## Model Evaluation

Each fold computes metrics such as Precision, Recall, and F1-Score. Loss and confusion matrices are also analyzed to assess convergence and identify errors.

## Model 1: Channel Fusion

Channel Fusion uses a convolutional network with spatial and channel attention mechanisms to classify EEG windows independently as "seizure" or "non-seizure."

### Architecture

1. **Input**: EEG windows with dimensions `(B, C, T, 1)`:
   - B: Batch size
   - C: Channels (21)
   - T: Temporal length (128)
   
2. **Convolutional Layers**:
   - Three sequential convolutional layers extract initial features, each followed by max pooling.

3. **Spatial Attention**:
   - Highlights important spatial regions using a 1x1 convolutional kernel.

4. **Channel Attention**:
   - Aggregates spatial information to prioritize significant channels.

5. **Fully Connected Layers**:
   - Features are flattened and passed through two fully connected layers to classify the windows.

### Advantages

- Integrates attention mechanisms to extract critical spatial and channel-specific features.
- Ensures robust generalization through hierarchical feature extraction.

## Model 2: LSTM

The LSTM model captures temporal relationships between windows by processing sequences of 10 consecutive windows.

### Data Preparation

- **Input**: The first 10,000 EEG windows per patient, flattened into a 1D representation.
- **Sequence Labeling**: A sequence is labeled as positive if any window within it is classified as a seizure; otherwise, it is labeled as negative.

### Architecture

1. **LSTM Layers**:
   - Two stacked LSTM layers with 256 hidden units and a dropout of 50%.
   
2. **Output Module**:
   - Fully connected layers classify the sequence as seizure or non-seizure.

### Advantages

- Captures temporal dependencies in EEG signals.
- Identifies sequential patterns that independent window analysis cannot.

## Model 3: Channel Fusion + LSTM

This model combines the spatial capabilities of Channel Fusion with the temporal insights of LSTM, creating a robust approach for detecting seizures.

### Data Preparation

- **Feature Extraction**: The pre-trained Channel Fusion model generates spatial features for each window, which are organized into sequences for the LSTM.
- **Labeling**: Follows the same sequence labeling strategy as the LSTM model.

### Architecture

1. **Channel Fusion**:
   - Pre-trained to extract spatial features from EEG windows.
   
2. **LSTM**:
   - Processes spatial features from Channel Fusion to learn temporal dependencies.

### Advantages

- Combines spatial and temporal strengths for improved accuracy.
- Reuses the pre-trained Channel Fusion model, reducing training time and ensuring consistent performance.

# Experimental Design

## 3.1 Dataset Description
The project utilized the publicly available **CHB-MIT dataset**, which includes EEG recordings from 24 patients, sampled at 256 Hz using a 21-channel EEG device. The dataset annotates the start and end of epileptic seizures for each patient. 

Key preprocessing steps included:
- **Resampling**: Data resampled to 128 Hz.
- **Windowing**: Divided into 1-second windows, resulting in matrices of dimensions `[N_windows, 21, 128]`.
- **Labeling**: Each window is classified as:
  - `0`: Normal (non-seizure).
  - `1`: Ictal (within a seizure period).

The dataset is structured into:
1. **NPZ Files**: Contain EEG window arrays.
2. **PARQUET Files**: Include metadata for each window, such as class and file identifiers.

This dataset enables the development of both **population-level** and **personalized models**, providing structured data for training and validation.

## 3.2 Experiments
The experiments evaluate three models: **Channel Fusion**, **LSTM**, and a **combined model** (Channel Fusion + LSTM). Each model is tested under two approaches:

### Population Approach
- **Cross-Validation**: 24 folds, with one patient per fold used for testing, while the remaining patients are for training.
- **Objective**: Assess the model’s ability to generalize across patients.

### Personalized Approach
- **Cross-Validation**: 10 folds, with data split into training and test sets from the same patients.
- **Objective**: Train and validate models using combined data from all patients.

### 3.2.1 Channel Fusion
The Channel Fusion model leverages convolutional layers with spatial and channel attention mechanisms to improve feature extraction from EEG windows.

- **Challenges**: Initial tests revealed overfitting issues, resolved by introducing attention mechanisms and optimizing hyperparameters.
- **Final Configuration**:
  - Optimizer: Adam (learning rate: 0.0005).
  - Batch Size: 64.
  - Dropout: 0.9 to reduce overfitting.

### 3.2.2 LSTM
The LSTM model captures temporal relationships in EEG windows, processing sequences of 10 consecutive windows.

- **Experiments**:
  - **Without Overlap**: Non-overlapping sequences, reducing memory usage but sacrificing temporal continuity.
  - **With Overlap**: Overlapping sequences (e.g., advancing by one window), improving temporal context at the cost of higher computational demand.

- **Final Configuration**:
  - Optimizer: Adam (learning rate: 0.001).
  - Batch Size: 512.
  - Hidden Layers: 2 LSTM layers with 256 units each.

### 3.2.3 Channel Fusion + LSTM
This combined model integrates Channel Fusion’s spatial feature extraction with LSTM’s temporal capabilities.

- **Data Preparation**:
  - EEG windows are processed through the pre-trained Channel Fusion model.
  - The resulting features are organized into sequences for the LSTM.

- **Challenges**:
  - Memory optimization issues were resolved by restructuring data loading, significantly reducing resource consumption.

- **Final Configuration**:
  - The Channel Fusion model remains pre-trained.
  - LSTM maintains the same settings as in the standalone model.

## 3.3 Metrics
To evaluate model performance, the following metrics were used:

1. **Precision**: Measures the proportion of correctly identified positives.
   - \( P = \frac{TP}{TP + FP} \)

2. **Recall**: Evaluates the proportion of actual positives detected.
   - \( R = \frac{TP}{TP + FN} \)

3. **F1-Score**: Balances Precision and Recall.
   - \( F1 = 2 \cdot \frac{P \cdot R}{P + R} \)

Additional metrics include confusion matrices and boxplots to identify variability in patient-specific results under the **Population Approach**. These analyses help detect potential biases and inform strategies for model improvement.

# Results Summary

This section presents the evaluation results of the three proposed models: **Channel Fusion**, **LSTM**, and the **combined model (Channel Fusion + LSTM)**. Results are analyzed for both the **Population Approach** and the **Personalized Approach**, and the impact of different configurations is discussed.

## 4.1 Model 1: Channel Fusion

### Population Approach
- **Challenges**: The Channel Fusion model initially showed poor generalization, with significant overfitting and a lack of convergence in most folds. The model struggled to adapt to unseen patients, likely due to patient-specific differences in EEG signals.
- **Performance**: Precision, recall, and F1-score exhibited high variability, especially for seizure detection (Class 1). False positives and false negatives were frequent, limiting the model's reliability for generalizing across patients.

### Personalized Approach
- **Improvements**: Incorporating data from all patients during training resulted in better convergence, with stable loss curves and reduced overfitting. The model achieved balanced precision and recall for both seizure (Class 1) and non-seizure (Class 0) windows.
- **Performance**: Metrics improved significantly compared to the Population Approach, with reduced variability and higher averages (around 0.9 for all metrics).

## 4.2 Model 2: LSTM

### Population Approach
- **Temporal Dimension**: Adding temporal relationships through LSTM significantly enhanced generalization compared to Channel Fusion. Loss values stabilized around 0.1 in most folds, and false negatives decreased notably.
- **Performance**: Precision, recall, and F1-score showed reduced variability and higher values for both classes, reaching averages around 0.94 for Class 0 and 0.88 for Class 1.

### Personalized Approach
- **Convergence**: Loss values stabilized around 0.05, reflecting the model's ability to adapt to patient-specific data. False positives and negatives were minimized, and metrics were highly consistent across folds.
- **Performance**: Precision, recall, and F1-score for both classes were above 0.95, showcasing exceptional performance and reliability.

### Overlap vs. Non-Overlap
- **Outcome**: Including overlapping sequences did not significantly improve performance compared to non-overlapping sequences. Overlapping added computational overhead without providing substantial benefits, making non-overlapping sequences the preferred configuration.
- 
## 4.3 Model 3: Channel Fusion + LSTM

### Population Approach
- **Combining Strengths**: By leveraging Channel Fusion's spatial feature extraction and LSTM's temporal capabilities, this model achieved improved convergence and lower loss values (<0.15).
- **Performance**: Precision and recall were nearly perfect for both classes, with averages around 0.99 for Class 0 and 0.97 for Class 1. Variability across folds was lower than in individual models.

### Personalized Approach
- **Consistency**: The model achieved rapid convergence and stable loss curves, with minimal false positives and negatives. Metrics demonstrated high precision and recall, approaching 1.0 for both classes.
- **Performance**: This model exhibited the best overall results, with outstanding reliability and minimal variability across folds.

## 4.4 Model Comparisons

### Population Approach
- **Channel Fusion**: Adequate for Class 0 (non-seizure), but weak for Class 1 (seizure), with high false negatives and low F1-scores.
- **LSTM**: Improved metrics for both classes, especially Class 1, demonstrating the importance of temporal relationships.
- **Channel Fusion + LSTM**: Outperformed both models, achieving nearly perfect precision, recall, and F1-scores for both classes.

### Personalized Approach
- **Channel Fusion**: Showed better results than in the Population Approach but remained less effective than the other models.
- **LSTM**: Demonstrated strong consistency and significant improvements in seizure detection (Class 1).
- **Channel Fusion + LSTM**: Delivered the best performance, with nearly perfect metrics and minimal variability across folds.

# 5. Discussion and Conclusions

This work successfully explored the implementation and comparison of different models for detecting epileptic seizures from EEG signals. Initially, the **Channel Fusion** model demonstrated its ability to identify relevant spatial patterns through the incorporation of attention mechanisms. By introducing the **LSTM**, the temporal dimension was captured, significantly improving performance metrics and reducing errors.

The combination of these two approaches in the **Channel Fusion + LSTM** model leveraged the strengths of each: the spatial attention from Channel Fusion and the sequential processing capability of the LSTM. This resulted in outstanding overall performance, both in the **Population Approach** and the **Personalized Approach**, establishing it as the most robust and consistent method. The results validate the multimodal approach for critical clinical applications, as it nearly minimizes a critical metric like recall.

In conclusion, the integration of spatial and temporal attention represents a significant advancement in the automated detection of epileptic seizures.



