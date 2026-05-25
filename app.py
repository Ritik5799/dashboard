import pandas as pd
import lightgbm as lgb
import joblib

from sklearn.model_selection import train_test_split

# LOAD DATA
train_transaction = pd.read_csv(
    "train_transaction.csv"
)

train_identity = pd.read_csv(
    "train_identity.csv"
)

# MERGE
data = train_transaction.merge(
    train_identity,
    how="left",
    on="TransactionID"
)

# TARGET
TARGET = "isFraud"

# PREPROCESSING
for col in data.columns:

    if data[col].dtype == "object":

        data[col] = data[col].fillna(
            "Unknown"
        )

        data[col] = data[col].astype(
            "category"
        ).cat.codes

    else:

        data[col] = data[col].fillna(
            data[col].median()
        )

# FEATURES
X = data.drop(columns=[TARGET])

y = data[TARGET]

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# MODEL
model = lgb.LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

# SAVE MODEL
joblib.dump(
    model,
    "model.pkl"
)

print("model.pkl saved successfully")