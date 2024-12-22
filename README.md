---
#  Système d’Irrigation Intelligent et Autonome

##  Description

Ce projet vise à développer un système d’irrigation intelligent, autonome et économe en eau, spécialement conçu pour les **zones désertiques**  par exemple la région d’El Oued en Algérie. Grâce à l'intégration de **modèles prédictifs LSTM** et d’un **agent d’apprentissage par renforcement profond (DQN)**, le système adapte en temps réel l’irrigation en fonction des besoins réels des cultures (ici, la tomate) et des conditions climatiques locales.
---
## Objectifs du Projet

- Optimiser la gestion de l’eau dans des zones à stress hydrique élevé.
- Améliorer les rendements agricoles en personnalisant l’irrigation.
- Réduire les coûts d’irrigation par une gestion intelligente des ressources.
- Intégrer des technologies d’**IA**, de simulation **agro-climatique**, et de **capteurs IoT**.

---

## Technologies et Méthodologies Utilisées

- **Deep Learning** : LSTM pour la prédiction du SWTD (soil water content) et du rendement.
- **Reinforcement Learning** : Deep Q-Network (DQN) pour la décision optimale d’irrigation.
- **Simulations agronomiques** : DSSAT avec le module CROPGRO-Tomato.
- **Sources de données** : NASA POWER API pour les données climatiques.
- **Prétraitement** : Z-Score, normalisation, PCA.
- **Évaluation** : RMSE, R² pour la précision des modèles.

---

## Modèles Implémentés

### LSTM 1 : Prédiction SWTD

- Entrées : Température, humidité, précipitations, évapotranspiration.
- Fenêtre temporelle : 4 à 7 jours.
- Sortie : Teneur en eau du sol le lendemain.
- R² obtenu : > 0.90

### LSTM 2 : Prédiction du rendement

- Entrées : Climat, irrigation historique, SWTD.
- Période : Saison agricole complète.
- Sortie : Estimation du rendement en kg/ha.
- R² obtenu : ≈ 0.91

### Agent DQN : Optimisation des volumes d’irrigation (en cours d'implementation)

- Actions possibles : 12 niveaux (0–60 mm/jour)
- Récompense basée sur : rendement économique - coût de l’eau

---

## Spécificités Régionales

- **Localisation** : Wilaya d’El Oued, climat saharien extrême.
- **Culture ciblée** : Tomate (variétés locales adaptées aux fortes chaleurs).
- **Irrigation** : Pivots agricoles mobiles automatisés.

---

## Auteurs

- **Yazi Lynda Mellissa**
- **Benmachiche Khaled**

---

## Implémentation Modulaire

La logique des notebooks a été refactorisée en modules réutilisables:

- `scripts/train_lstm.py`: point d'entrée CLI pour les expériences.
- `configs/lstm1_swtd.yaml`: config LSTM 1 (prédiction SWTD).
- `configs/lstm2_cwad.yaml`: config LSTM 2 (prédiction rendement/CWAD).
- `src/irrigation_system/preprocessing.py`: nettoyage, normalisation, fenêtrage temporel.
- `src/irrigation_system/model_builder.py`: construction paramétrique du BiLSTM.
- `src/irrigation_system/evaluation.py`: métriques de régression et analyse d'erreurs.
- `src/irrigation_system/lstm1_swtd.py`: pipeline SWTD (Z-score + StandardScaler).
- `src/irrigation_system/lstm2_cwad.py`: pipeline CWAD (MinMaxScaler indépendant).
- `src/irrigation_system/common.py`: utilitaires (seeds, I/O, timestamps).
- `src/irrigation_system/config.py`: chargement des configs YAML.

## Installation

Environnement Python 3.12 recommandé:

```bash
python3.12 -m venv .venv312
. .venv312/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Installation with UV

```bash
uv sync
```

Install with development tools:

```bash
uv sync --group dev
```

Run commands through UV:

```bash
uv run python scripts/train_lstm.py --config configs/lstm1_swtd.yaml
uv run pyright src/ scripts/
uv run ruff check src/ scripts/
```

## Lancer les expériences

LSTM 1 (SWTD):

```bash
.venv312/bin/python scripts/train_lstm.py --config configs/lstm1_swtd.yaml
```

LSTM 2 (CWAD):

```bash
.venv312/bin/python scripts/train_lstm.py --config configs/lstm2_cwad.yaml
```

## Service API (deploy)

Entrypoint FastAPI: [scripts/api.py](scripts/api.py)

Run local API:

```bash
.venv312/bin/uvicorn scripts.api:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

Prediction example:

```bash
curl -X POST http://localhost:8000/predict \
	-H "Content-Type: application/json" \
	-d '{"sequence": [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]}'
```

Container deployment:

- Dockerfile: [Dockerfile](Dockerfile)

```bash
docker build -t irrigation-api .
docker run --rm -p 8000:8000 -e MODEL_PATH=models/latest.keras irrigation-api
```

## Artifacts générés

- logs finaux: `artifacts/lstm/logs/*.final.log`
- registre d'expériences: `artifacts/lstm/logs/experiment_runs.csv`
- métriques détaillées: `artifacts/lstm/metrics/*.json`
- prédictions test: `artifacts/lstm/predictions/*.csv`
- checkpoints modèles: `models/*.keras`
