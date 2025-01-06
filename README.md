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
3. [Results](#results)
4. [Future Improvements](#future-improvements)
5. [Conclusion](#conclusion)
6. [How to Run the Code](#how-to-run-the-code)
7. [References](#references)

---

## Introduction

This project, conducted as part of the **Ma513 - Hands-on Machine Learning for Cybersecurity** module, explores the use of deep learning models for Named Entity Recognition (NER) in unstructured textual data. NER is a subfield of Natural Language Processing (NLP) that identifies and classifies specific entities such as locations, organizations, and people automatically.

Our team chose to utilize a pre-trained BERT model, fine-tuned to maximize accuracy in a cybersecurity context.

---

## General Methodology

### **Data Exploration and Preprocessing**

The dataset is sourced from SemEval-2018 Task 8 ("SecureNLP") and is formatted in JSON Lines. Each entry contains:

- `unique_id`: Unique identifier for the sentence.
- `tokens`: List of tokens (strings) forming the text.
- `ner_tags`: Named Entity Recognition tags following the IOB2 convention.

Example Entry:

```json
{
   "unique_id": 4775,
   "tokens": [
      "This", 
      "collects", 
      ":", 
      "Collected", 
      "data", 
      "will", 
      "be", 
      "uploaded", 
      "to", 
      "a", 
      "DynDNS", 
      "domain", 
      "currently", 
      "hosted", 
      "on", 
      "a", 
      "US", 
      "webhosting", 
      "service", 
      "."
   ],
   "ner_tags": [
      "B-Entity", 
      "B-Action", 
      "O", 
      "B-Entity", 
      "I-Entity", 
      "O", 
      "B-Action", 
      "I-Action", 
      "B-Modifier", 
      "B-Entity", 
      "I-Entity", 
      "I-Entity", 
      "I-Entity", 
      "I-Entity", 
      "I-Entity", 
      "I-Entity", 
      "I-Entity", 
      "I-Entity", 
      "I-Entity", 
      "O"
   ]
}
```

### **Model Selection**

- Evaluation of various approaches (rule-based systems, machine learning, deep learning).
- Adoption of **BERT**, a bidirectional neural network model, known for its outstanding performance in complex NLP tasks.

### **Training and Fine-Tuning**

- Loading and configuring hyperparameters for the BERT model.
- Fine-tuning the last layers of the network to improve results.
- Optimizing weights using the **AdamW** optimizer.

### **Result Evaluation**

- Performance measurement using metrics such as the **F1 score**.
- Detailed analysis of results by class: Action, Entity, Modifier.

---

## Results

- **Global F1 Score:** 27.85%
  - Highlights the need for improvement, particularly in terms of precision (23.88%) and recall (33.39%).
- **Class-wise Results:**
  - **Action:** F1 = 57.21%
  - **Modifier:** F1 = 56.03%
  - **Entity:** F1 = 11.39%

While the Action and Modifier classes showed acceptable performance, the Entity class proved particularly challenging, likely due to its contextual complexity.

---

## Future Improvements

Integrating a hybrid system combining BERT with advanced techniques, such as **semantic role labeling**, could enhance performance by leveraging contextual relationships more effectively.
Remove punctuation.

---

## Conclusion

Despite technical challenges, our project demonstrated the effectiveness of BERT in an NER task for cybersecurity. Using a large model with fine-tuning significantly improved results. However, exploring hybrid approaches offers promising potential for future advancements.

---

## How to Run the Code

### **Prerequisites**

1. **Install Python 3.8 or later.**
   - Ensure `pip` is installed.
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
   ```
   And install the version of torch, torchvision and torchaudio with your following version of CUDA. If you don't have CUDA on your 
3. **Clone the repository and navigate to the project directory:**
   ```bash
   git clone https://github.com/Rsky-20/ma513_HandsOnMachineLearning_for_cybersecurity
   cd ma513_HandsOnMachineLearning_for_cybersecurity
   ```
4. **Prepare the dataset:**
   - Place the JSON Lines files (`NER-TRAINING.jsonlines`, `NER-VALIDATION.jsonlines`, `NER-TESTING.jsonlines`) in the `./data/` directory.

### **Run the Training Script**


### **Outputs**

- The fine-tuned model is saved in the `./ner-model/` directory.
- Predictions are saved as JSON Lines in the `./data/` directory.

---

## References

1. [Named Entity Recognition - Geekflare](https://geekflare.com/fr/named-entity-recognition/)
2. [Named Entity Recognition - Klippa](https://www.klippa.com/fr/blog/informations/reconnaissance-dentites-nommees/)
