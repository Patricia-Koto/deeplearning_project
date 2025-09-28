# 🌍 Classification de Scènes Naturelles  

## 📌 Description du Projet  
Ce projet a pour objectif de classifier des **images de scènes naturelles** (bâtiments, forêts, glaciers, montagnes, mers, rues) à l’aide d’un modèle de **Deep Learning en transfert learning**.  

L’application développée avec **Streamlit** permet de :  
- Charger un modèle entraîné (`transfer_best.keras`).  
- Importer une ou plusieurs images en même temps.  
- Obtenir les prédictions du modèle avec les probabilités associées (Top-k).  
- Visualiser les résultats directement dans le navigateur.  

---

## 🗂 Jeu de Données  
- Dataset utilisé : **Intel Image Classification (Kaggle)**  
- ~25 000 images réparties en **6 classes** :  
  - 🏙️ `buildings`  
  - 🌳 `forest`  
  - ❄️ `glacier`  
  - 🏔️ `mountain`  
  - 🌊 `sea`  
  - 🚦 `street`  

Lien : [Intel Image Classification – Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

---

## ⚙️ Installation  

### 1. Cloner le projet  
```bash
git clone https://github.com/ton-compte/scenes-classifier.git
cd scenes-classifier
```

### 2. Créer un environnement virtuel (optionnel mais recommandé)  
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Installer les dépendances  
Avec le fichier `requirements.txt` :  
```bash
pip install -r requirements.txt
```

⚡ Ou version minimale :  
```bash
pip install streamlit tensorflow keras pillow matplotlib numpy
```

---

## ▶️ Utilisation  

Lancer l’application :  
```bash
streamlit run app.py
```

Une interface s’ouvre dans le navigateur (par défaut : [http://localhost:8501](http://localhost:8501)).  

### Fonctionnalités principales :  
- **Uploader plusieurs images** en une fois.  
- Choisir le **Top-k** (nombre de classes les plus probables à afficher).  
- Visualiser l’image originale + les probabilités sous forme de texte et barre de progression.  

---

## 📊 Modèle utilisé  
- **Transfer Learning** basé sur MobileNetV2 (pré-entraîné sur ImageNet).  
- Fine-tuning sur les 6 classes du dataset Intel Scenes.  
- Sauvegarde du meilleur modèle via `ModelCheckpoint` → `transfer_best.keras`.  

---

## 📈 Résultats  
- Accuracy d’entraînement/validation suivie par courbes `loss` et `accuracy`.  
- Évaluation finale avec précision, rappel, F1-score et matrice de confusion (voir le rapport PDF `Projet  Classification de Scènes Naturelles.pdf`).  

---

## 📂 Structure du projet  
```
.
├── app.py                     # Application Streamlit
├── transfer_best.keras        # Modèle entraîné sauvegardé
├── requirements.txt           # Dépendances
├── Projet Classification...pdf# Rapport de projet
├── script_final.ipynb         # Notebook d'entraînement
└── README.md                  # Documentation
```

---

## 🚀 Améliorations possibles  
- Ajouter une **visualisation Grad-CAM** pour comprendre les zones d’attention du modèle.  
- Permettre la **capture caméra en direct**.  
- Ajouter l’export des prédictions en CSV.  
- Tester d’autres architectures pré-entraînées (ResNet50, EfficientNet).  

---

## 👩‍💻 Auteurs  
Projet réalisé dans le cadre d’un **projet académique en Deep Learning** par :  
- **Patricia Koto Ngbanga**  
