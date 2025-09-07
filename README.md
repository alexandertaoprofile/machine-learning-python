This repository contains a collection of Python projects showcasing the application of machine learning and simulation techniques to different problem domains.
The main focus is on developing and experimenting with algorithms for prediction, simulation, and decision-making, with an emphasis on skills relevant to computational chemistry/materials science machine learning workflows (e.g., model building, feature engineering, SHAP interpretability, Monte Carlo simulation).

Contents
1. 🏇 Race Win Rate Prediction (XGBoost + SHAP)
Built predictive models (XGBoost, Lasso, Random Forest) to analyze horse-racing/game-style competition outcomes.
Feature set includes skill IDs, support card IDs, attributes (speed, stamina, power, etc.).
Techniques: feature mapping via JSON, train/test splits across domains, grouped importance ranking (skills, cards, strategies).

2. 🎲 Mingchao Monopoly-style Game Simulation
Implemented a Monte Carlo simulation framework for a turn-based board game.
Encodes complex skill rules, Probability-based stacking and movement interactions.
Conditional effects (e.g., movement doubled, escape from stacks, comeback mechanics).
Simulates thousands of games to estimate win rates and ranking distributions.

Skills Demonstrated
Machine Learning: XGBoost, SHAP, feature engineering, model evaluation.

Simulation & Probability: Monte Carlo methods, stochastic modeling.

Software Engineering: Object-oriented programming (OOP), modular code design.

Visualization: Matplotlib/SHAP plots for model interpretability

Data Handling: JSON mappings, CSV parsing, multi-dataset validation.
