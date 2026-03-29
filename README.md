🚗 Used Car Selling Price Prediction
A complete machine learning pipeline to predict the selling price of used cars using multiple regression models — with both regression and classification-style evaluation metrics.

📁 Project Structure
Predict_sellingprice/
├── data/
│   └── Cardetails.csv          # Raw dataset (add manually)
├── car_price_prediction.py     # Main ML pipeline script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── eda_scatter.png             # Generated: scatter plots
├── price_distribution.png      # Generated: price histogram
├── model_comparison.png        # Generated: R² and MAE bar charts
├── actual_vs_predicted.png     # Generated: best model predictions
└── feature_importance.png      # Generated: top features

📊 Dataset

File: Cardetails.csv — place it inside the data/ folder
Rows: ~8,000 used car listings
Target column: selling_price (in ₹)

FeatureDescriptionnameCar model nameyearManufacturing yearselling_pricePrice sold at (target)km_drivenKilometers drivenfuelFuel type (Petrol/Diesel/CNG/LPG)seller_typeIndividual / Dealer / TrustmarktransmissionManual / Automaticowner1st, 2nd, 3rd, 4th+ ownermileageFuel efficiency (kmpl)engineEngine displacement (CC)max_powerMax power (bhp)torqueTorque (dropped — complex units)seatsNumber of seats

⚙️ Pipeline Steps

Load data — reads Cardetails.csv
Feature engineering — extracts numeric values from mileage, engine, max_power; derives car_age and brand
Missing values — median fill for numeric, mode fill for categorical
Deduplication — drops exact duplicate rows
Outlier removal — IQR-based per numeric column
Encoding — ordinal for owner, one-hot for other categoricals
EDA plots — scatter plots and price distribution histogram
Train/test split — 80/20, random_state=42
Scaling — StandardScaler fitted on train only (no data leakage)
Hyperparameter tuning — RandomizedSearchCV with 5-fold CV for tree models
Evaluation — regression metrics + classification-style metrics via binning


🤖 Models Trained
ModelNotesRandom ForestTuned with RandomizedSearchCVGradient BoostingTuned with RandomizedSearchCVDecision TreeTuned with RandomizedSearchCVLinear RegressionBaseline linear modelLasso RegressionL1 regularizationRidge RegressionL2 regularizationKNN RegressorDistance-based, uses scaled features

📈 Evaluation Metrics
Regression: R², MAE, RMSE
Classification-style (continuous prices binned into 5 ranges): Accuracy, Precision, Recall, F1-Score

🚀 Getting Started
1. Clone the repository
bashgit clone https://github.com/siyapatel696/Predict_sellingprice.git
cd Predict_sellingprice
2. Create and activate a virtual environment
bash# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
bashpip install -r requirements.txt
4. Add the dataset
Place Cardetails.csv inside the data/ folder:
Predict_sellingprice/
└── data/
    └── Cardetails.csv
5. Run the pipeline
bashpython car_price_prediction.py
Or open in Jupyter:
bashjupyter notebook

📦 Key Improvements Over Baseline
IssueFixmileage, engine, max_power were stringsExtracted numeric values with regexyear used rawConverted to car_age = 2024 - yearname dropped rawExtracted brand (first word) for lower cardinalityfillna(..., inplace=True) FutureWarningReplaced with df[col] = df[col].fillna(...)Scaler fit on full data (leakage)Scaler fitted only on training setowner one-hot encodedOrdinal encoded (has natural order)KNN on unscaled featuresKNN now receives scaled featuresLasso not convergingAdded max_iter=5000 and proper alphaSeaborn palette deprecationUpdated to hue= pattern

📋 Requirements

Python 3.9+
See requirements.txt for full dependency list


👩‍💻 Author
Siya Patel
GitHub: @siyapatel696