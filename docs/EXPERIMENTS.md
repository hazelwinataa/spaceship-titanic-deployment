# Kaggle classification experiment — recorded evidence

## Dataset and EDA

Original notebook `notebooks/spaceship_titanic_experiments.ipynb` loads `train.csv` and records 8,693 training passengers with 14 initial columns including `Transported`. Distribution: 4,378 True and 4,315 False. The notebook visualizes target balance, transport response across HomePlanet/CryoSleep/Destination/VIP and histograms of Age plus onboard spending.

## Feature engineering

Cabin → Deck, Cabin_num, Side; PassengerId → Group / Group_size / Solo; Name → FirstName / LastName / Family_size; onboard spending → TotalSpending, HasSpending/NoSpending, per-category spending ratios; Age → Age_group; missingness flags for Age/CryoSleep. The saved output reports 34 engineered dataframe columns. The model uses 27 selected columns.

## Results and evaluation boundaries

| Model | Baseline held-out validation accuracy | MLflow baseline 5-fold CV mean | MLflow tuned 5-fold CV mean |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.76998 | 0.78810 | 0.79305 |
| XGBoost | 0.79931 | 0.80524 | **0.81261** |
| LightGBM | **0.81081** | 0.81088 | 0.81146 |

The notebook baseline comparison uses a stratified 80/20 hold-out (6,954 train / 1,739 validation). Later sections use five-fold CV on the labeled dataset and 30 Optuna trials per model. Models and metrics are logged to an MLflow experiment. XGBoost is registered as version 3 under `@champion` in saved notebook output.

**Kaggle:** The uploaded screenshot shows public score **0.79401**, from a successful submission. The notebook submission cell calls `lr_final.predict(X_test)` (Logistic Regression), which differs from the MLflow Registry champion. The specific notebook revision behind the shown Kaggle submission is not independently verified.

## Limitations / important implementation notes

- The original notebook preprocesses and encodes the full training data before cross-validation, which may bias estimates; refactor with train-fitted preprocessing within folds.
- Separate LabelEncoder fitting on `test.csv` risks mismatched categorical numeric mappings; persist train-fitted encoders/ColumnTransformer.
- Customer/family/group frequencies derived globally need leakage consideration; related passengers should be grouped in validation where warranted.
- Do not claim the Kaggle public score is the MLflow CV accuracy, or that a registered champion was the submitted model.
- No independent full rerun was performed when assembling this public documentation.
