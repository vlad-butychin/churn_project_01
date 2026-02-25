# %%
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score

# %%
df = pd.read_csv(r"C:\Users\admin\Desktop\churn_project\data\Churn.csv",sep=',')
df.head()
df.info()
df.isna().sum()

# %%
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# %%
df_encoded = df.drop('customerID', axis=1)
num_col = df_encoded.select_dtypes(include=[np.number]).columns
cat_col = df_encoded.select_dtypes(include=object).columns

df[num_col] = df[num_col].fillna(df[num_col].median())
for col in cat_col:
    df[col] = df[col].fillna(df[col].mode()[0])

# %%
bynary_cat = [col for col in cat_col if len(df[col].unique()) == 2]
multi_cat = [col for col in cat_col if len(df[col].unique()) > 2]

label_encoders = {}
for i in bynary_cat:
    le = LabelEncoder()
    df_encoded[i] = le.fit_transform(df_encoded[i])
    label_encoders[i] = le

df_encoded = pd.get_dummies(df_encoded, columns=multi_cat, drop_first=True)

# %%
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)

scaler = StandardScaler()
X_train[num_col] = scaler.fit_transform(X_train[num_col])
X_test[num_col] = scaler.transform(X_test[num_col])

# %%
lr = LogisticRegression(random_state=42, max_iter=1000,class_weight='balanced')
lr.fit(X_train, y_train)
proba = lr.predict_proba(X_test)[:,1]
custom_pred = (proba > 0.4).astype(int)
print(classification_report(y_test, custom_pred))

# %%
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print(classification_report(y_test, rf_pred))

# %%
print("LogReg ROC-AUC:", roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]))
print("RF ROC-AUC:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

# %%
coef_df = pd.DataFrame({
    'feature': X.columns,
    'coefficient': lr.coef_[0]
}).sort_values('coefficient', key=abs, ascending=False)
coef_df.head(6)

# %%
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
feature_importance.head(6)

# %%
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

fpr, tpr, thresholds = roc_curve(y_test, proba)

plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.show()


