
---

#  Système d’Irrigation Intelligent et Autonome

##  Description

Ce projet vise à développer un système d’irrigation intelligent, autonome et économe en eau, spécialement conçu pour les **zones désertiques**  par exemple la région d’El Oued en Algérie. Grâce à l'intégration de **modèles prédictifs LSTM** et d’un **agent d’apprentissage par renforcement profond (DQN)**, le système adapte en temps réel l’irrigation en fonction des besoins réels des cultures (ici, la tomate) et des conditions climatiques locales.

---

##  Objectifs du Projet

- Optimiser la gestion de l’eau dans des zones à stress hydrique élevé.
- Améliorer les rendements agricoles en personnalisant l’irrigation.
- Réduire les coûts d’irrigation par une gestion intelligente des ressources.
- Intégrer des technologies d’**IA**, de simulation **agro-climatique**, et de **capteurs IoT**.

---

##  Technologies et Méthodologies Utilisées

- **Deep Learning** : LSTM pour la prédiction du SWTD (soil water content) et du rendement.
- **Reinforcement Learning** : Deep Q-Network (DQN) pour la décision optimale d’irrigation.
- **Simulations agronomiques** : DSSAT avec le module CROPGRO-Tomato.
- **Sources de données** : NASA POWER API pour les données climatiques.
- **Prétraitement** : Z-Score, normalisation, PCA.
- **Évaluation** : RMSE, R² pour la précision des modèles.

---

##  Modèles Implémentés

###  LSTM 1 : Prédiction SWTD
- Entrées : Température, humidité, précipitations, évapotranspiration.
- Fenêtre temporelle : 4 à 7 jours.
- Sortie : Teneur en eau du sol le lendemain.
- R² obtenu : > 0.90

###  LSTM 2 : Prédiction du rendement
- Entrées : Climat, irrigation historique, SWTD.
- Période : Saison agricole complète.
- Sortie : Estimation du rendement en kg/ha.
- R² obtenu : ≈ 0.91

###  Agent DQN : Optimisation des volumes d’irrigation (en cours d'implementation)
- Actions possibles : 12 niveaux (0–60 mm/jour)
- Récompense basée sur : rendement économique - coût de l’eau

---

## Spécificités Régionales

- **Localisation** : Wilaya d’El Oued, climat saharien extrême.
- **Culture ciblée** : Tomate (variétés locales adaptées aux fortes chaleurs).
- **Irrigation** : Pivots agricoles mobiles automatisés.

---
##  Auteurs
- **Yazi Lynda Mellissa**  
- **Benmachiche Khaled**  


---
