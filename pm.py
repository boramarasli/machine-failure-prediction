#predictive maintanence - my own rebuild 
# given data: ai4i2020.csv from UCI, see the folder

import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import average_precision_score
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                      cross_val_score, train_test_split)
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

#Step 1 - look at the data 
df = pd.read_csv("ai4i2020.csv")
print(df.shape)
print(df.head())

failures = df["Machine failure"].sum()
print(f"Failures: {failures} of {len(df)} ({failures/len(df):.1%})")

#a model that always says no failure: 
always_no = 1 - df["Machine failure"].mean()
print(f"Always saying 'no failure' is right {always_no:.1%} of the time- and catches 0 failures.")

#s2 clean and split
df= df.drop(columns=["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"])

df["Type"] = df["Type"].map({"L": 0, "M": 1, "H": 2})
y = df["Machine failure"]
X = df.drop(columns=["Machine failure"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, stratify = y, random_state = 42)

#s3 first model- raw sensors only
cv = StratifiedKFold(n_splits= 5, shuffle = True, random_state =42)
forest = RandomForestClassifier(n_estimators = 300, random_state = 42, n_jobs = -1)

raw = cross_val_score(forest, X_train, y_train, cv= cv, scoring= "average_precision")
print(f"Random forest, raw sensord: average precision {raw.mean():.2f}")

#s4 adding physics
def add_physics(data): 
    data = data.copy()
    omega = data["Rotational speed [rpm]"] * 2 * np.pi / 60  #rpm -> rad/s
    data["power_W"] = data["Torque [Nm]"] * omega 
    data["temp_diff_K"] = data ["Process temperature [K]"] - data["Air temperature [K]"]
    data["strain"] = data["Tool wear [min]"] * data["Torque [Nm]"]
    return data 

X_train_phys = add_physics(X_train)
X_test_phys = add_physics(X_test)

phys = cross_val_score(forest, X_train_phys, y_train, cv= cv, scoring = "average_precision")
print(f"Random forest, with physics: average precision {phys.mean():.2f}")

#s5 a simpler model on the same inputs 

logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter = 1000))
log = cross_val_score(logreg, X_train_phys, y_train, cv= cv, scoring="average_precision")
print(f"Logistic regression, physics:average precision {log.mean():.2f}")

#s6 alarm level by ost then the final test - only once. 
COST_MISSED = 5000 #assumed: a breakdown no one saw coming 
COST_FALSE = 200 #assumed: someone checks a machine that was fine 

def cost(y_true, probs, threshold): 
    alarm = probs >= threshold 
    missed = ((y_true == 1) & ~alarm).sum()
    false = ((y_true == 0) & alarm).sum()
    return int(COST_MISSED * missed + COST_FALSE * false)

#choose alarm level on the training data only
train_probs = cross_val_predict(forest, X_train_phys, y_train, cv= cv, 
                                method = "predict_proba")[:, 1]

thresholds = np.arange(0.01, 1.00, 0.01)
costs = [cost(y_train.values, train_probs, t) for t in thresholds]
best = thresholds[np.argmin(costs )]
print(f"Cheapest alarm level (chosen on training data): {best:.2f}")

forest.fit(X_train_phys, y_train)
test_probs = forest.predict_proba(X_test_phys)[:, 1]
print(f"Test set: average precision {average_precision_score(y_test, test_probs):.2f}")

for t in [0.5, best]: 
    alarm = test_probs >= t
    caught = ((y_test == 1)& alarm).sum()
    false = ((y_test == 0)& alarm).sum()
    print(f"Alarm at {t:.2f}: caught {caught} of {y_test.sum()} failures, " 
          f"{false} false alarms, cost {cost(y_test.values, test_probs, t):,} EUR")
