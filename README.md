# 🎌 Anime Recommender System — Modèle Hybride (Content-Based + Collaborative)

Système de recommandation d’animés personnalisé basé sur le dataset **Anime 2023**, combinant **filtrage basé sur le contenu**, **filtrage collaboratif** et **similarité hybride** pour offrir des suggestions plus pertinentes.


## 🎯 **Description du Projet**

L’objectif de ce projet est de développer un moteur de recommandation d’animés performant en exploitant :

* ✔️ Les caractéristiques des animés (genres, studios, score, popularité…)
* ✔️ Le comportement des utilisateurs (notes, préférences)
* ✔️ Une approche hybride combinant :

  * **Content-Based Filtering**
  * **Collaborative Filtering**
  * **KNN sur matrice de similarité**
  * **Matrice utilisateur-anime**

Ce système permet d’obtenir des recommandations beaucoup plus précises que les systèmes simples basés uniquement sur les genres.


## 📂 **Dataset**

Dataset : **Anime 2023 Dataset**
Contient des milliers d’animés avec :

| Colonne    | Description          |
| ---------- | -------------------- |
| Title      | Nom de l’animé       |
| Genre      | Liste des genres     |
| Type       | TV, Movie, OVA, etc. |
| Episodes   | Nombre d’épisodes    |
| Score      | Note moyenne         |
| Popularity | Index de popularité  |
| Studio     | Studio principal     |
| Synopsis   | Résumé textuel       |


## 🧹 **Prétraitement des Données**

### ✔️ Nettoyage

* Suppression des lignes manquantes essentielles
* Normalisation des colonnes scores/popularité/épisodes
* Tokenization du synopsis (optionnel)

### ✔️ Encodage des genres

* Découpage multi-label
* Encodage via **MultiLabelBinarizer**

### ✔️ Encodage des studios / types

* **OneHotEncoder** avec gestion automatique des catégories inconnues

### ✔️ Matrice finale

Combinaison de :

* Features numériques
* Genres encodés
* Studios / types encodés
* Embeddings textuels (optionnel)


## 🤖 **Approches de Recommandation**

### 🔷 **1. Content-Based Filtering**

Basé sur :

* Genres
* Studio
* Format (TV, Movie…)
* Popularité / Score

Utilisation :

```
Cosine Similarity  
or  
Euclidean Distance
```

### 🔷 **2. Collaborative Filtering**

Basé sur les notes des utilisateurs :

* Création d’une **user-anime matrix**
* Modèle :
  ✔️ KNN (User-based filtering)
  ✔️ Similarité Cosine

### 🔷 **3. Modèle Hybride**

Combinaison pondérée :

```
Recommandation finale = α * Content + (1-α) * Collaborative
```


## 📊 **Exemple d’Utilisation**

```python
def recommend(anime_title, n=5):
    idx = anime_df[anime_df["Title"] == anime_title].index[0]
    distances, indices = knn_model.kneighbors(final_matrix[idx], n_neighbors=n+1)
    return anime_df.iloc[indices[0][1:]]
```


## 🚀 **Technologies Utilisées**

* Python 3
* pandas / numpy
* scikit-learn
* matplotlib / seaborn
* Jupyter Notebook
* Sentence-Transformers (optionnel pour embeddings)


## ▶️ **Lancer le Projet**

### 1. Cloner le projet

```bash
git clone https://github.com/username/anime-recommender.git
cd anime-recommender
```

### 2. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3. Lancer l’entraînement

```bash
jupyter notebook Anime_Recommender.ipynb
```

### 4. Tester une recommandation

```python
recommend("Attack on Titan")
```


## ✨ **Améliorations Futures**

* Ajout d’un modèle **BERT / Transformer** pour analyser les synopsis
* Pondération dynamique (score, genres, popularité…)
* Déploiement d’une interface via **Streamlit** ou **FastAPI**
* Modèle LightFM (hybrid collaborative/content)
* Clustering des animés via PCA / t-SNE


## 👤 **Auteur**

**Alex Alkhatib**
Projet Machine Learning — Anime Recommendation System


## 📄 Licence
MIT License
Copyright (c) 2025 Alex Alkhatib
