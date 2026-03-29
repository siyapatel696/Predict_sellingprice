import re, warnings, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import randint, uniform
warnings.filterwarnings("ignore")

df = pd.read_csv("data/Cardetails.csv")

def extract_numeric(s):
    return s.astype(str).str.extract(r"([\d.]+)")[0].astype(float)

df["mileage"]   = extract_numeric(df["mileage"])
df["engine"]    = extract_numeric(df["engine"])
df["max_power"] = extract_numeric(df["max_power"])
df.drop(columns=["torque"], inplace=True, errors="ignore")
df["car_age"] = 2024 - df["year"]
df.drop(columns=["year"], inplace=True)
df["brand"] = df["name"].str.split().str[0]
df.drop(columns=["name"], inplace=True)

for col in df.select_dtypes(include=["float64","int64"]).columns:
    df[col] = df[col].fillna(df[col].median())
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop_duplicates()

def remove_outliers(df, col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    return df[(df[col] >= Q1-1.5*IQR) & (df[col] <= Q3+1.5*IQR)]

for col in df.select_dtypes(include=["float64","int64"]).columns:
    df = remove_outliers(df, col)

owner_order = {"First Owner":1,"Second Owner":2,"Third Owner":3,
               "Fourth & Above Owner":4,"Test Drive Car":0}
df["owner"] = df["owner"].map(owner_order).fillna(2).astype(int)
df = pd.get_dummies(df, columns=["fuel","seller_type","transmission","brand"], drop_first=True)

X = df.drop("selling_price", axis=1)
y = df["selling_price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X_test.columns)

rf_search = RandomizedSearchCV(RandomForestRegressor(random_state=42),
    {"n_estimators":randint(100,300),"max_depth":randint(5,20),
     "min_samples_split":randint(2,10),"min_samples_leaf":randint(1,5)},
    n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42)
gb_search = RandomizedSearchCV(GradientBoostingRegressor(random_state=42),
    {"n_estimators":randint(100,300),"max_depth":randint(3,8),
     "learning_rate":uniform(0.01,0.2),"min_samples_split":randint(2,10)},
    n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42)
dt_search = RandomizedSearchCV(DecisionTreeRegressor(random_state=42),
    {"max_depth":randint(5,20),"min_samples_split":randint(2,10),"min_samples_leaf":randint(1,5)},
    n_iter=10, cv=3, scoring="r2", n_jobs=-1, random_state=42)

print("Fitting models...")
rf_search.fit(X_train, y_train)
gb_search.fit(X_train, y_train)
dt_search.fit(X_train, y_train)

bins = np.linspace(y.min(), y.max(), 6)
y_test_b = np.digitize(y_test, bins)

models = {
    "Random Forest":     (rf_search.best_estimator_, X_train,   X_test),
    "Gradient Boosting": (gb_search.best_estimator_, X_train,   X_test),
    "Decision Tree":     (dt_search.best_estimator_, X_train,   X_test),
    "Linear Regression": (LinearRegression(),         X_train_s, X_test_s),
    "Lasso":             (Lasso(alpha=10,max_iter=5000), X_train_s, X_test_s),
    "Ridge":             (Ridge(alpha=10),             X_train_s, X_test_s),
    "KNN":               (KNeighborsRegressor(n_neighbors=7), X_train_s, X_test_s),
}

results = []
for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    yp = model.predict(Xte)
    yp_b = np.digitize(yp, bins)
    results.append({"Model":name,
        "R2":round(r2_score(y_test,yp),4),
        "MAE":round(mean_absolute_error(y_test,yp),2),
        "RMSE":round(np.sqrt(mean_squared_error(y_test,yp)),2),
        "Accuracy":round(accuracy_score(y_test_b,yp_b),4),
        "F1":round(f1_score(y_test_b,yp_b,average="weighted",zero_division=0),4)})

res = pd.DataFrame(results).sort_values("R2", ascending=False)
print("\n=== Model Results ===")
print(res.to_string(index=False))

fig, axes = plt.subplots(1,2,figsize=(12,5))
axes[0].barh(res["Model"], res["R2"], color="steelblue")
axes[0].set_title("R² Score"); axes[0].set_xlim(0,1)
axes[1].barh(res["Model"], res["MAE"], color="coral")
axes[1].set_title("MAE")
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=100)
plt.show()
print("\nDone! Best model:", res.iloc[0]["Model"], "| R²:", res.iloc[0]["R2"])