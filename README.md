![Description of the image](.\assets/NER_img.png)


# Hands-on Machine Learning for Cybersecurity: Named Entity Recognition (NER)

## Introduction

This project, conducted as part of the **Ma513 - Hands-on Machine Learning for Cybersecurity** module, explores the use of deep learning models for Named Entity Recognition (NER) in unstructured textual data. NER is a subfield of Natural Language Processing (NLP) that identifies and classifies specific entities such as locations, organizations, and people automatically.

Our team (Valentin DESFORGES, Sylvain LAGARENNE, Pierre VAUDRY) chose to utilize a pre-trained BERT model, fine-tuned to maximize accuracy in a cybersecurity context.

## General Methodology

1. **Data Exploration and Preprocessing**
   - Analysis of existing data structured in the IOB2 format, which categorizes each word based on its role within an entity.
   - Data preparation to ensure optimal compatibility with advanced NLP models.

2. **Model Selection**
   - Evaluation of various approaches (rule-based systems, machine learning, deep learning).
   - Adoption of BERT, a bidirectional neural network model, known for its outstanding performance in complex NLP tasks.

3. **Training and Fine-Tuning**
   - Loading and configuring hyperparameters for the BERT model.
   - Fine-tuning the last layers of the network to improve results.
   - Optimizing weights using the AdamW optimizer.

4. **Result Evaluation**
   - Performance measurement using metrics such as the F1 score.
   - Detailed analysis of results by class: Action, Entity, Modifier.

## Results

- **Global F1 Score:** 27.85%
  - Highlights the need for improvement, particularly in terms of precision (23.88%) and recall (33.39%).
- **Class-wise Results:**
  - Action: F1 = 57.21%
  - Modifier: F1 = 56.03%
  - Entity: F1 = 11.39%

While the Action and Modifier classes showed acceptable performance, the Entity class proved particularly challenging, likely due to its contextual complexity.

## Future Improvements

Integrating a hybrid system combining BERT with advanced techniques, such as semantic role labeling, could enhance performance by leveraging contextual relationships more effectively.

## Conclusion

Despite technical challenges, our project demonstrated the effectiveness of BERT in an NER task for cybersecurity. Using a large model with fine-tuning significantly improved results. However, exploring hybrid approaches offers promising potential for future advancements.

## References

1. [Named Entity Recognition - Geekflare](https://geekflare.com/fr/named-entity-recognition/)
2. [Named Entity Recognition - Klippa](https://www.klippa.com/fr/blog/informations/reconnaissance-dentites-nommees/)

---

For any questions or contributions, feel free to contact the project team.
