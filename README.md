# Analyse de Sentiment

## 1. Présentation Générale

### 1.1. Objectif

La société *Air Paradis* souhaite détecter les éventuels *Bad Buzz* afin d'anticiper ou de préparer une réponse rapide.
Dans ce cadre, nous développons un outil permettant d'analyser des tweets et de prédire leur sentiment (positif ou négatif).

### 1.2. Domaine d'étude

Nous utilisons des réseaux de neurones profonds, spécialement dans le domaine du *Natural Language Processing* (NLP).
Les données sont composées de tweets en langue anglaise.

#### Exemples de tweets classifiables

- **"Great day out"** (*Super journée.*) → Sentiment **positif**
- **"Goodness I am tired... Ugh lack of sleep is so bad for me."** (*Mon dieu, je suis épuisé... Le manque de sommeil est mauvais pour moi.*) → Sentiment **négatif**
- **"@TheSilentCoyote what about the podcast?"** → Difficile à classifier

Les défis de cette classification incluent :
- **Orthographe et syntaxe approximatives**
- **Implicite** (humour, ironie, sarcasme)
- **Analyse du contexte**

### 1.3. Principes et évaluation

Le jeu de données est divisé en deux :
- **Jeu d'entraînement** pour configurer nos modèles
- **Jeu de test** pour évaluer les performances

L'équilibre parfait entre tweets positifs et négatifs garantit une évaluation fiable.

Nous explorons plusieurs approches et comparons les modèles selon plusieurs métriques :
- **Précision**, **Recall**, **Spécificité**, **Accuracy**, **F1-score**
- **Courbe ROC & AUC**, **Matrice de confusion**
- **Temps d'entraînement**

### 1.4. Préparation des données

Les tweets subissent une **tokenisation** avant d'être transformés de trois manières :
- Texte brut (*Raw*)
- **Lemmatisation**
- **Stemmatisation**

### 1.5. Modélisation

Trois approches sont testées :
1. **Modèle simple** : Algorithmes de *Machine Learning* rapide
2. **Modèle avancé** : *Deep Learning* avec *Word Embedding*
3. **Modèle BERT** : *Transfer Learning*

---

## 2. Analyse Exploratoire

Les données sont issues de [Sentiment140 (Kaggle)](https://www.kaggle.com/datasets/kazanova/sentiment140).
Elles comprennent 1,6 million de tweets avec 6 caractéristiques, mais nous retenons seulement :
- **Text** : Contenu du tweet
- **Target** : Sentiment (0 = négatif, 4 = positif)

#### Répartition des sentiments

**Variable cible parfaitement équilibrée** :
- 800 000 tweets positifs
- 800 000 tweets négatifs

---

## 3. Prétraitement (*Preprocessing*)

### 3.1. Substitutions appliquées aux tweets

1. Remplacement des **URLs** par `'<url>'`
2. Remplacement des **@usernames** par `'<user>'`
3. Réduction des **lettres répétées** (`'Heyyyy' → 'Heyy'`)
4. Remplacement des **émojis** par des équivalents textuels
5. Expansion des **contractions** (`"can't" → "cannot"`)
6. Suppression des **caractères spéciaux**

### 3.2. Tokenisation

Utilisation de `word_tokenize` (NLTK) suivie de lemmatisation/stemmatisation.

---

## 4. Modélisation

Nous utilisons un échantillon de 200 000 tweets (données équilibrées).

### 4.1. Modèle simple (*Machine Learning*)

**Algorithmes testés :**
- **Régression Logistique**
- **Naive Bayes**
- **Support Vector Classifier (SVC)**

#### Résultats

| Modèle | Accuracy (sans preprocessing) |
|---------|--------------------------|
| Logistic Regression | **80.28%** |
| SVC | 79.69% |
| Naive Bayes | 78.58% |

Modèle retenu : **Régression Logistique (C=2, 1000 itérations)**

**Performances :**
- **Accuracy** : 80.28%
- **ROC AUC** : 88.38%
- **Temps d'entraînement** : 30 sec

---

### 4.2. Modèle avancé (*Deep Learning*)

**Word Embeddings testés :**
- **FastText**, **GloVe**, **Word2Vec**

**Avec et sans LSTM**

#### Résultats

| Embedding | Sans LSTM | Avec LSTM |
|-----------|----------|-----------|
| **GloVe** | 78.16% | **79.30%** |
| Word2Vec | 75.51% | 76.63% |
| FastText | 77.56% | 78.98% |

**Modèle retenu :** GloVe + LSTM

---

### 4.3. Modèle BERT (*Transfer Learning*)

Utilisation de **bert-base-uncased** avec *fine-tuning*.

#### Résultats

| Modèle | Accuracy | ROC AUC |
|---------|---------|---------|
| **BERT** | **84.25%** | **90.21%** |

---

## Conclusion

- **Le modèle BERT offre les meilleures performances**.
- **LSTM + GloVe est une alternative valable** si les ressources sont limitées.
- **Les modèles classiques (Logistic Regression) sont rapides mais moins précis**.

---

Ce projet montre l'importance du choix de modèle et du prétraitement pour l'analyse de sentiment ! 🚀

