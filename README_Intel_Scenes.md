# Classification de Scènes Naturelles — CNN & Transfer Learning (Keras/TensorFlow)

Ce projet propose un pipeline complet pour classifier des images de scènes naturelles (dataset **Intel Image Classification**) en utilisant :
- un **CNN baseline** entraîné depuis zéro, et
- un modèle **MobileNetV2 en transfer learning** avec data augmentation et callbacks,
ainsi que des outils d’**évaluation**, d’**explicabilité Grad‑CAM**, et des fonctions de prédiction **sur une image ou un dossier complet**.

> Notebook principal : `Intel_Scenes_Classification.ipynb`

---

## 1) Jeu de données

- Source : [Intel Image Classification (Kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- Classes (ordre alphabétique) : `buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`
- Organisation des dossiers attendue :

```
data/
 ├─ seg_train/seg_train/      # 6 sous-dossiers (un par classe)
 └─ seg_test/seg_test/        # utilisé comme validation dans ce projet
```
> Le dossier `seg_pred` (non étiqueté) n’est pas utilisé.

Vérifiez/éditez la variable `BASE_DIR` dans le notebook si nécessaire.

---

## 2) Environnement

- Python 3.9+ (Anaconda recommandé)
- TensorFlow 2.12+ / 2.15+
- Keras (inclus dans TensorFlow)
- NumPy, Matplotlib, scikit‑learn, Pillow

Installation minimale :
```bash
pip install tensorflow numpy matplotlib scikit-learn pillow
```

GPU recommandé mais non obligatoire.

---

## 3) Structure du notebook

1. **Configuration & chemins** : création de `train_ds` et `val_ds` via `image_dataset_from_directory` (`IMG_SIZE=(150,150)`, `BATCH_SIZE=32`).
2. **EDA (optionnel)** : statistiques (nombre d’images, exemples par classe, tailles d’images, visualisation des augmentations).
3. **Optimisation tf.data** : `cache().prefetch(AUTOTUNE)` pour accélérer l’entraînement.
4. **Data Augmentation & Normalisation** : `RandomFlip`, `RandomRotation`, `RandomZoom`, `RandomContrast`.
5. **Modèle baseline CNN** : petits blocs Conv‑BN‑ReLU + GlobalAveragePooling + Dense softmax.
6. **Entraînement (baseline)** : callbacks (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`).
7. **Évaluation** : rapport de classification et **matrice de confusion** sur `val_ds`.
8. **Grad‑CAM** : génération de cartes de chaleur pour expliquer les prédictions.
9. **Transfer Learning MobileNetV2** : backbone gelé + tête de classification spécifique. Le prétraitement est **inclus dans le modèle**, donc inutile à l’inférence.
10. **Prédictions** :
    - `predict_and_show(model, img_path, class_names)` → une image avec top‑k prédictions
    - `predict_folder_and_show(model, folder, class_names, csv_path=...)` → un dossier complet, avec affichage en grille + export CSV

---

## 4) Exécution

1. Télécharger et extraire le dataset Kaggle sous `data/` comme indiqué ci-dessus.
2. Modifier dans le notebook :
   ```python
   BASE_DIR = r"C:\Users\<VOUS>\...\deeplearning_project\data"
   TRAIN_DIR = os.path.join(BASE_DIR, "seg_train", "seg_train")
   VAL_DIR   = os.path.join(BASE_DIR, "seg_test",  "seg_test")
   IMG_SIZE   = (150, 150)
   BATCH_SIZE = 32
   SEED       = 42
   ```
3. Lancer les cellules pour entraîner soit le **baseline CNN**, soit directement le **MobileNetV2 transfer learning**.
4. Après l’entraînement, exécuter la cellule d’évaluation :
   ```python
   eval_ds = val_ds
   # Affiche classification_report et matrice de confusion
   ```
5. Prédiction sur une image :
   ```python
   predict_and_show(transfer, r"C:\chemin\vers\image.jpg", class_names, img_size=(150,150))
   ```
6. Prédiction sur un dossier complet + export CSV :
   ```python
   _ = predict_folder_and_show(transfer,
                               r"C:\chemin\vers\dossier",
                               class_names,
                               img_size=(150,150),
                               cols=4,
                               csv_path="preds_dossier.csv")
   ```

> ⚠️ Important : avec MobileNetV2, utiliser `preprocess_fn = lambda x: x` et garder `img_size=(150,150)`.

---

## 5) Conseils et options

- **Callbacks** : déjà configurés (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`). Ajuster `patience` et `factor` si besoin.
- **Classes déséquilibrées** : possibilité de calculer `class_weight` depuis `train_ds` et de le passer à `model.fit(...)`.
- **Fine‑tuning** : défiger les derniers blocs MobileNetV2 après quelques époques (learning rate réduit, ex. `1e-5`).
- **Grad‑CAM** : utile pour comprendre les zones utilisées par le modèle.
- **Export** : sauvegarder `class_names` dans `class_names.json` pour garder le même mapping à l’inférence.

---

## 6) Fichiers du repo

- `Intel_Scenes_Classification.ipynb` — notebook principal
- `class_names.json` — mapping des classes sauvegardé (généré après entraînement)
- `preds_dossier.csv` — fichier CSV de prédictions exporté (créé après utilisation des helpers)

---

## 7) Remerciements

- Dataset Intel Image Classification (Kaggle)
- Keras/TensorFlow (MobileNetV2)
- Inspirations : notebooks communautaires Kaggle

---

## 8) Problèmes fréquents

- **Erreur chemins Windows** : utiliser `r"C:\chemin\vers\img.jpg"` ou bien des `/` classiques.
- **“Aucune image trouvée”** : donner le chemin d’un dossier (et non un fichier) à `predict_folder_and_show`.
- **Toutes les prédictions dans une seule classe** : vérifier que le prétraitement est identique entre entraînement et inférence et que `class_names` correspond bien à l’ordre utilisé pendant l’entraînement.

---

Bon entraînement ! 🚀
