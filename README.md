# covid19_RNN
#  Prédiction COVID-19 avec des Réseaux de Neurones Récurrents

> **Projet Deep Learning — Master 1 Intelligence Artificielle**  
> Université de Béjaïa — 2025/2026

---

##  Description

Ce projet implémente et compare trois architectures de réseaux de neurones récurrents améliorés pour la prédiction des cas cumulatifs de COVID-19 sur plusieurs pays :

- **GRU amélioré** — Gated Recurrent Unit avec connexion résiduelle
- **LSTM amélioré** — Long Short-Term Memory avec connexion résiduelle
- **BiLSTM amélioré** — LSTM bidirectionnel avec connexion résiduelle

Les modèles sont entraînés et évalués sur les données de **5 pays** : France 🇫🇷, Chine 🇨🇳, Royaume-Uni 🇬🇧, Australie 🇦🇺 et Pays-Bas 🇳🇱.

---

##  Structure du projet

```
COVID19_RNN/
├── COVID19_RNN___.ipynb        # Notebook principal

├── README.md                   # Ce fichier

```

---

##  Prérequis

### Environnement Python
- Python ≥ 3.8

### Dépendances

```bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn scikit-learn
```

Ou via conda :

```bash
conda install pytorch torchvision -c pytorch
conda install numpy pandas matplotlib seaborn scikit-learn
```

---

##  Lancement

1. Cloner ou télécharger le projet
2. Installer les dépendances
3. Ouvrir et exécuter le notebook cellule par cellule :

```bash
jupyter notebook COVID19_RNN___.ipynb
```

> **Note :** Le notebook télécharge automatiquement les données depuis le dépôt GitHub de Johns Hopkins University au démarrage. Une connexion internet est requise.

---

##  Données

**Source :** [Johns Hopkins CSSE COVID-19 Dataset](https://github.com/CSSEGISandData/COVID-19)

Les trois séries temporelles utilisées sont :
- `time_series_covid19_confirmed_global.csv` — cas confirmés
- `time_series_covid19_deaths_global.csv` — décès
- `time_series_covid19_recovered_global.csv` — guérisons

### Prétraitement
- Agrégation par pays (somme des provinces)
- Fusion des trois datasets par pays et par date
- Extraction des cas journaliers par différenciation
- Normalisation MinMax sur les cas cumulatifs confirmés
- Création de séquences par fenêtre glissante de **60 jours**
- Partitionnement : **70% train / 15% validation / 15% test**

---

##  Architecture des modèles

Tous les modèles partagent le même schéma avec **connexion résiduelle** :

$$\hat{y}_{t+1} = x_t + \alpha \cdot f_\theta(x_{t-L+1}, \ldots, x_t)$$

où $\alpha = 0.05$ est un facteur d'atténuation pour stabiliser l'entraînement.

### GRU amélioré
| Couche | Détails |
|--------|---------|
| GRU | 128 neurones, 2 couches, dropout 0.2 |
| FC | Linear(128→64) → ReLU → Dropout(0.1) → Linear(64→1) |

### LSTM amélioré
| Couche | Détails |
|--------|---------|
| LSTM | 128 neurones, 2 couches, dropout 0.2 |
| FC | Linear(128→64) → ReLU → Dropout(0.1) → Linear(64→1) |

### BiLSTM amélioré
| Couche | Détails |
|--------|---------|
| BiLSTM | 128 neurones, 2 couches, bidirectionnel, dropout 0.2 |
| FC | Linear(256→64) → ReLU → Dropout(0.1) → Linear(64→1) |

> Le BiLSTM double la taille de la sortie (256 = 128 × 2) en raison du caractère bidirectionnel.

---

##  Entraînement

Tous les modèles utilisent la même procédure d'entraînement :

| Hyperparamètre | Valeur |
|----------------|--------|
| Optimiseur | AdamW (weight_decay=1e-5) |
| Learning rate | 0.001 |
| Scheduler | ReduceLROnPlateau (patience=30, factor=0.5) |
| Gradient clipping | max_norm=1.0 |
| Early stopping | patience=100 |
| Epochs max | 700 |
| Loss | MSE |
| Graine aléatoire | 42 |

Le meilleur état du modèle (meilleur val loss) est sauvegardé et rechargé en fin d'entraînement.

---

##  Métriques d'évaluation

Deux niveaux de métriques sont calculés :

| Métrique | Description |
|----------|-------------|
| MAE normalisé | Erreur absolue moyenne sur l'espace [0, 1] |
| RMSE normalisé | Racine de l'erreur quadratique moyenne normalisée |
| MAE réel | MAE en nombre de cas cumulés |
| RMSE réel | RMSE en nombre de cas cumulés |
| R² | Coefficient de détermination (calculé sur les valeurs réelles) |

---

##  Pays analysés

Le notebook applique le même pipeline complet à chacun des pays suivants :

| Pays | Variable cible |
|------|---------------|
| France 🇫🇷 | Cas cumulatifs confirmés |
| Chine 🇨🇳 | Cas cumulatifs confirmés (somme des provinces) |
| Royaume-Uni 🇬🇧 | Cas cumulatifs confirmés |
| Australie 🇦🇺 | Cas cumulatifs confirmés |
| Pays-Bas 🇳🇱 | Cas cumulatifs confirmés |

---

##  Reproductibilité

Les graines aléatoires sont fixées avant chaque entraînement pour assurer la reproductibilité :

```python
torch.manual_seed(42)
np.random.seed(42)
```

---

##  Plan du notebook

| Section | Contenu |
|---------|---------|
| 1–3 | Imports, chargement et fusion des données Johns Hopkins |
| 4 | Exploration et visualisation (France, Italie, États-Unis + Top 10) |
| 5 | Prétraitement France (normalisation, séquences, split) |
| 6–7 | Fonctions d'évaluation et d'entraînement communes |
| 8–10 | Définition et entraînement des 3 modèles (France) |
| 11–16 | Courbes d'apprentissage, évaluation et visualisation (France) |
| 17–27 | Pipeline complet — Chine |
| 28–35 | Pipeline complet — Royaume-Uni |
| 34–41 | Pipeline complet — Australie |
| 40–47 | Pipeline complet — Pays-Bas |
| Finale | Résumé global — tous les pays |

---

##  Auteur

Projet réalisé dans le cadre du **Master 1 Intelligence Artificielle**  
Université de Béjaïa — Année universitaire 2025/2026
