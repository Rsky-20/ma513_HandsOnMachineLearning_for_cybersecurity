from transformers import AutoConfig, BertForTokenClassification, BertTokenizerFast, Trainer, TrainingArguments, pipeline
from datasets import DatasetDict, Dataset, Features, Sequence, ClassLabel, Value
import torch
import json
import numpy as np
import evaluate
import os
import warnings
import time
import psutil

# Désactiver les warnings inutiles
warnings.filterwarnings("ignore")

# Enregistrer l'heure de début
start_time = time.time()

# Configurer le dispositif (GPU ou CPU)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Définir les variables globales
TRAINING_FILE = "./data/NER-TRAINING.jsonlines"
VALIDATION_FILE = "./data/NER-VALIDATION.jsonlines"
TESTING_FILE = "./data/NER-TESTING.jsonlines"
MODEL_PATH = "./bert-large-ner-model"
LABELS = ["B-Action", "B-Entity", "B-Modifier", "I-Action", "I-Entity", "I-Modifier", "O"]

# Charger le tokenizer
TOKENIZER = BertTokenizerFast.from_pretrained("dslim/bert-large-NER")

def load_and_prepare_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    # Vérifier et ajouter la clé 'ner_tags' si elle est manquante
    for example in data:
        if "ner_tags" not in example:
            example["ner_tags"] = ["O"] * len(example["tokens"])  # Valeur par défaut
    return data

def convert_to_dataset_with_labels(data_section, labels):
    has_ner_tags = all("ner_tags" in example for example in data_section)

    features = Features({
        "id": Value("int64"),
        "tokens": Sequence(Value("string")),
    })
    if has_ner_tags:
        features["ner_tags"] = Sequence(ClassLabel(names=labels))
    
    dataset_dict = {
        "id": [example["unique_id"] for example in data_section],
        "tokens": [example["tokens"] for example in data_section],
    }
    if has_ner_tags:
        dataset_dict["ner_tags"] = [example["ner_tags"] for example in data_section]

    return Dataset.from_dict(dataset_dict, features=features)

def tokenize_and_align_labels(examples):
    tokenized_inputs = TOKENIZER(
        examples["tokens"], 
        truncation=True, 
        padding="max_length", 
        max_length=131,  # Longueur maximale des séquences
        is_split_into_words=True
    )
    if "ner_tags" in examples:
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
    return tokenized_inputs



# Charger les données brutes
train_data = load_and_prepare_data(TRAINING_FILE)
validation_data = load_and_prepare_data(VALIDATION_FILE)
test_data = load_and_prepare_data(TESTING_FILE)

# Liste des labels
ner_labels = ["B-Action", "B-Entity", "B-Modifier", "I-Action", "I-Entity", "I-Modifier", "O"]

# Créer un DatasetDict avec des labels
ner_data = DatasetDict({
    "train": convert_to_dataset_with_labels(train_data, ner_labels),
    "validation": convert_to_dataset_with_labels(validation_data, ner_labels),
    "test": convert_to_dataset_with_labels(test_data, ner_labels)
})

tokenized_datasets = ner_data.map(tokenize_and_align_labels, batched=True)

# Créer une nouvelle configuration adaptée à vos labels
config = AutoConfig.from_pretrained(
    "dslim/bert-large-NER",
    num_labels=len(LABELS),  # Votre propre nombre de labels
    id2label={i: label for i, label in enumerate(LABELS)},
    label2id={label: i for i, label in enumerate(LABELS)},
)

# Charger le modèle avec la nouvelle configuration
model = BertForTokenClassification.from_pretrained(
    "dslim/bert-large-NER",
    config=config,
    ignore_mismatched_sizes=True
).to(DEVICE)


# Configurer les arguments d'entraînement
args = TrainingArguments(
    output_dir=MODEL_PATH,            # Chemin pour sauvegarder le modèle
    evaluation_strategy="epoch",     # Évaluer à la fin de chaque époque
    save_strategy="epoch",           # Sauvegarder à la fin de chaque époque
    learning_rate=2e-5,              # Taux d'apprentissage
    per_device_train_batch_size=32,  # Taille du batch pour l'entraînement
    per_device_eval_batch_size=32,   # Taille du batch pour l'évaluation
    num_train_epochs=10,             # Nombre d'époques
    logging_dir='./logs',            # Répertoire pour les logs
    logging_steps=10,                # Logs après chaque 10 étapes
    fp16=True,                       # Activer l'entraînement en précision mixte (16 bits)
    load_best_model_at_end=True,     # Charger le meilleur modèle à la fin
    metric_for_best_model="f1",      # Critère pour sélectionner le meilleur modèle
    greater_is_better=True,          # Une valeur plus élevée de F1 est meilleure
    save_total_limit=2,              # Limiter le nombre de modèles sauvegardés
    weight_decay=0.01,
)


# Définir les métriques
metric = evaluate.load("seqeval")

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=2)
    true_predictions = [
        [LABELS[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [LABELS[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

# Entraîner le modèle
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=TOKENIZER,
    compute_metrics=compute_metrics,
)

trainer.train()

# Charger le modèle pour prédiction
nlp = pipeline("ner", model=model, tokenizer=TOKENIZER, device=0 if DEVICE == "cuda" else -1)

# Labels définis dans votre modèle
ner_labels = ["B-Action", "B-Entity", "B-Modifier", "I-Action", "I-Entity", "I-Modifier", "O"]

# Initialiser le pipeline NER
nlp = pipeline("ner", model=model.to(DEVICE), tokenizer=TOKENIZER, device=0 if DEVICE == 'cuda' else -1)

# Fonction pour convertir les indices en labels
def convert_indices_to_labels(indices, label_list):
    """
    Convertit une liste d'indices en étiquettes à l'aide de la liste de labels.
    """
    return [label_list[int(idx)] for idx in indices]  # Conversion explicite en int

# Fonction pour générer les prédictions et écrire dans un fichier JSONlines
def predict_and_save(ner_dataset, output_file):
    results = []
    for example in ner_dataset:
        # Effectuer les prédictions sur les tokens
        tokens = example["tokens"]
        ner_results = nlp(" ".join(tokens))
        
        # Initialiser les ner_tags prédits
        ner_tags_predicted = ["O"] * len(tokens)
        
        for ner_result in ner_results:
            label = ner_result["entity"]
            
            # Trouver le mot correspondant au résultat NER
            word = ner_result["word"]
            try:
                token_idx = tokens.index(word)
                ner_tags_predicted[token_idx] = label
            except ValueError:
                # Si le mot ne correspond pas, continuez
                continue

        # Ajouter le résultat au format JSONlines
        results.append({
            "unique_id": int(example["id"]),  # Conversion explicite en int
            "tokens": tokens,
            "ner_tags": convert_indices_to_labels(example.get("ner_tags", []), ner_labels),  # Conversion des indices
            "ner_tags_predicted": ner_tags_predicted
        })
    
    # Écrire les résultats dans le fichier JSONlines
    with open(output_file, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item) + "\n")
    print(f"Prédictions enregistrées dans {output_file}")

# Effectuer les prédictions pour le jeu de test
predict_and_save(ner_data["test"], "./data/NER-TESTING-PREDICTED.jsonlines")

# Effectuer les prédictions pour le jeu de validation
predict_and_save(ner_data["validation"], "./data/NER-VALIDATION-PREDICTED.jsonlines")

# Fonction pour évaluer les prédictions
def evaluate_predictions(validation_file, label_list):
    """
    Évalue les prédictions en comparant les ner_tags_predicted aux ner_tags.
    """
    # Charger les prédictions
    with open(validation_file, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f]

    metric = evaluate.load("seqeval")
    all_predictions = []
    all_references = []

    for example in examples:
        if "ner_tags" in example and "ner_tags_predicted" in example:
            references = example["ner_tags"]
            predictions = example["ner_tags_predicted"]

            all_predictions.append(predictions)
            all_references.append(references)

    # Calcul des métriques
    results = metric.compute(predictions=all_predictions, references=all_references)
    print("Résultats d'évaluation :")
    print(json.dumps(results, indent=2, default=str))  # Ajout de default=str
    return results

# Évaluer sur le jeu de validation
validation_results = evaluate_predictions("./data/NER-VALIDATION-PREDICTED.jsonlines", ner_labels)

