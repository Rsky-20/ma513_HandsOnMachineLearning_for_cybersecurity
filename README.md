# Hands-on Machine Learning for Cybersecurity: Named Entity Recognition (NER)

|                       | **Details**                                            |
|-----------------------|--------------------------------------------------------|
| **Date of Creation**  | 27.10.2024                                             |
| **Team**              | Valentin DESFORGES, Sylvain LAGARENNE, Pierre VAUDRY   |
| **Version**           | v6.01.2025                                             |
| **Python Version**    | Python 3.12.2                                          |
| **CUDA Version**      | 12.4                                                   |
|**Graphique Card Used**| Nvidia RTX 4060 Ti                                     |

---

## Table of Contents
1. [Introduction](#introduction)
2. [General Methodology](#general-methodology)
3. [Analysis of the Database](#analysis-of-the-database)
4. [Pre-processing Steps](#pre-processing-steps)
5. [Model Selection and Experimental Settings](#model-selection-and-experimental-settings)
6. [Results](#results)
7. [Future Improvements](#future-improvements)
8. [Conclusion](#conclusion)
9. [How to Run the Code](#how-to-run-the-code)
10. [References](#references)

---

## Introduction

This project, conducted as part of the **Ma513 - Hands-on Machine Learning for Cybersecurity** module, explores the use of deep learning models for Named Entity Recognition (NER) in unstructured textual data. NER identifies and classifies specific entities such as locations, organizations, and people automatically.

Our team leveraged a pre-trained BERT model, fine-tuned to maximize accuracy in a cybersecurity context. The work also involved deep learning techniques for understanding semantic relationships within text.

---

## General Methodology

### Data Tokenization

- Tokenization transforms raw text into a format understandable by the model.
- Labels use the IOB2 format (Inside, Outside, Beginning).

### Classification

- Entities are classified into Action, Entity, and Modifier classes.
- Training involves splitting data into training, validation, and test sets.

### Validation

- Fine-tuning was performed, followed by model validation using F1 scores and other metrics.

---

## Analysis of the Database

- **Format**: JSON Lines with `unique_id`, `tokens`, and `ner_tags`.
- **Labeling Scheme**: IOB2 format for tagging entities.
- **Classes Observed**: Action, Entity, Modifier.
- **Observations**:
  - Sentences are tokenized and include punctuation.
  - Data is already prepared for training and testing.

---

## Pre-processing Steps

1. Convert input files to DataFrames for easier manipulation.
2. Use functions for loading, converting, and tokenizing data:
   - `load_and_prepare_data`
   - `convert_to_dataset_with_labels`
   - `tokenize_and_align_labels`
3. Tokenize datasets for efficient processing:
   ```python
   tokenized_datasets = ner_data.map(tokenize_and_align_labels, batched=True)
   ```

---

## Model Selection and Experimental Settings

### Model Selection

- Chosen Model: **BERT** (Bidirectional Encoder Representations from Transformers).
- Pre-trained model: `dslim/bert-large-NER`.
- Selected for its ability to understand both syntactic and semantic contexts.

### Hyperparameter Initialization

- **Learning Rate**: 2 × 10⁻⁵
- **Batch Size**: 32
- **Epochs**: 10

Hyperparameter tuning showed minimal effects on performance.

### Metrics

- **Precision**: Correctly predicted entities / Total predicted entities.
- **Recall**: Correctly predicted entities / Total actual entities.
- **F1 Score**: Harmonic mean of precision and recall.
- **Accuracy**: Correctly classified tokens / Total tokens.

---

## Results

### Global Results

- **F1 Score**: 27.85%
- **Precision**: 23.88%
- **Recall**: 33.39%

### Class-wise Results

- **Action**: F1 = 57.21%
- **Modifier**: F1 = 56.03%
- **Entity**: F1 = 11.39%

The Entity class performed poorly, indicating challenges with contextual understanding.

---

## Future Improvements

- Integrate a **hybrid model** combining BERT with semantic role labeling techniques to better capture contextual relationships.
- Explore methods for handling punctuation effectively.

---

## Conclusion

The project demonstrated the utility of BERT for NER tasks in cybersecurity, achieving moderate success with Action and Modifier classes. Future work should focus on hybrid approaches and enhancing contextual learning.

---

## How to Run the Code

### Prerequisites

1. **Install Python 3.8 or later**.
   - Ensure `pip` is installed.
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
   ```
   Replace `cuXXX` with your CUDA version. Use CPU-compatible versions if CUDA is unavailable.
3. **Clone the repository**:
   ```bash
   git clone https://github.com/Rsky-20/ma513_HandsOnMachineLearning_for_cybersecurity
   cd ma513_HandsOnMachineLearning_for_cybersecurity
   ```
4. **Prepare the dataset**:
   Place the JSON Lines files (`NER-TRAINING.jsonlines`, `NER-VALIDATION.jsonlines`, `NER-TESTING.jsonlines`) in the `./data/` directory.

### Run the Training Script

```bash
python train_ner.py
```

### Outputs

- Fine-tuned model: `./ner-model/`
- Predictions: `./data/`

---

## References

1. [Named Entity Recognition - Geekflare](https://geekflare.com/fr/named-entity-recognition/)
2. [Named Entity Recognition - Klippa](https://www.klippa.com/fr/blog/informations/reconnaissance-dentites-nommees/)
