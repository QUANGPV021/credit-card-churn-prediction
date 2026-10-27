import json
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)

# Set style
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.unicode_minus"] = False
sns.set_theme(style="whitegrid", palette="muted")

os.makedirs("charts", exist_ok=True)

# 1. Load dataset
df_raw = pd.read_csv("BankChurners.csv")
print(f"Raw shape: {df_raw.shape}")

# Drop CLIENTNUM and the 2 Naive Bayes artifact columns recommended by dataset author
drop_cols = [c for c in df_raw.columns if c.startswith("Naive_Bayes_") or c == "CLIENTNUM"]
df = df_raw.drop(columns=drop_cols).copy()
print(f"Cleaned shape: {df.shape}")

# Create binary target: 1 = Attrited Customer (Rời bỏ), 0 = Existing Customer (Hiện hữu)
df["Churn"] = (df["Attrition_Flag"] == "Attrited Customer").astype(int)
df["Trang_Thai"] = df["Attrition_Flag"].map({
    "Existing Customer": "Khách hàng hiện hữu (0)",
    "Attrited Customer": "Khách hàng rời bỏ (1)"
})

summary = {
    "total_rows": int(df.shape[0]),
    "total_cols_raw": int(df_raw.shape[1]),
    "total_cols_clean": int(df.shape[1] - 2), # 20 original cols (19 features + 1 target)
    "missing_values": int(df.isnull().sum().sum()),
    "churn_counts": df["Attrition_Flag"].value_counts().to_dict(),
    "churn_rate_pct": round(float(df["Churn"].mean() * 100), 2),
}

# --- CHART 1: Churn Distribution ---
fig, ax = plt.subplots(figsize=(7, 5))
counts = df["Trang_Thai"].value_counts()
colors = ["#2b5c8f", "#d95f02"]
bars = ax.bar(counts.index, counts.values, color=colors, width=0.5, edgecolor="black")
for bar in bars:
    h = bar.get_height()
    pct = h / len(df) * 100
    ax.text(bar.get_x() + bar.get_width()/2, h + 150, f"{h:,} ({pct:.2f}%)",
            ha="center", va="bottom", fontweight="bold", fontsize=11)
ax.set_title("Phân bố Trạng thái Khách hàng Thẻ tín dụng (N = 10,127)", fontsize=13, fontweight="bold", pad=15)
ax.set_ylabel("Số lượng khách hàng", fontsize=11)
ax.set_ylim(0, max(counts.values) * 1.15)
plt.tight_layout()
plt.savefig("charts/01_churn_distribution.png")
plt.close()

# --- CHART 2: Demographics vs Churn ---
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 2a. Age distribution
sns.histplot(data=df, x="Customer_Age", hue="Trang_Thai", kde=True, bins=25,
             palette=colors, ax=axes[0, 0], alpha=0.6)
axes[0, 0].set_title("Phân bố Độ tuổi theo Trạng thái Rời bỏ", fontweight="bold")
axes[0, 0].set_xlabel("Độ tuổi (Customer_Age)")
axes[0, 0].set_ylabel("Số lượng")

# 2b. Gender churn rate
gender_churn = df.groupby("Gender")["Churn"].mean() * 100
bars_g = axes[0, 1].bar(["Nữ (F)", "Nam (M)"], [gender_churn["F"], gender_churn["M"]],
                        color=["#e78ac3", "#66c2a5"], width=0.45, edgecolor="black")
for b in bars_g:
    axes[0, 1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.5, f"{b.get_height():.2f}%",
                    ha="center", fontweight="bold")
axes[0, 1].set_title("Tỷ lệ Rời bỏ theo Giới tính (%)", fontweight="bold")
axes[0, 1].set_ylabel("Tỷ lệ rời bỏ (%)")
axes[0, 1].set_ylim(0, 25)

# 2c. Income Category churn rate
inc_order = ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"]
inc_churn = df.groupby("Income_Category")["Churn"].mean().reindex(inc_order) * 100
sns.barplot(x=inc_churn.values, y=inc_churn.index, palette="Blues_r", ax=axes[1, 0], edgecolor="black")
for i, v in enumerate(inc_churn.values):
    axes[1, 0].text(v + 0.3, i, f"{v:.1f}%", va="center", fontweight="bold")
axes[1, 0].set_title("Tỷ lệ Rời bỏ theo Nhóm Thu nhập (%)", fontweight="bold")
axes[1, 0].set_xlabel("Tỷ lệ rời bỏ (%)")
axes[1, 0].set_xlim(0, 23)

# 2d. Education Level churn rate
edu_churn = (df.groupby("Education_Level")["Churn"].mean() * 100).sort_values(ascending=False)
sns.barplot(x=edu_churn.values, y=edu_churn.index, palette="Oranges_r", ax=axes[1, 1], edgecolor="black")
for i, v in enumerate(edu_churn.values):
    axes[1, 1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontweight="bold")
axes[1, 1].set_title("Tỷ lệ Rời bỏ theo Trình độ Học vấn (%)", fontweight="bold")
axes[1, 1].set_xlabel("Tỷ lệ rời bỏ (%)")
axes[1, 1].set_xlim(0, 26)

plt.tight_layout()
plt.savefig("charts/02_demographic_analysis.png")
plt.close()

# --- CHART 3: Transaction Behavior (Key Driver) ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

sns.kdeplot(data=df, x="Total_Trans_Ct", hue="Trang_Thai", fill=True, palette=colors, ax=axes[0], alpha=0.4, linewidth=2)
axes[0].axvline(55, color="red", linestyle="--", label="Ngưỡng rủi ro cao (< 55 giao dịch)")
axes[0].set_title("Phân bố Tổng số lần Giao dịch trong 12 tháng (Total_Trans_Ct)", fontweight="bold")
axes[0].set_xlabel("Tổng số lần giao dịch (12 tháng)")
axes[0].set_ylabel("Mật độ")
axes[0].legend()

sns.scatterplot(data=df.sample(2500, random_state=42), x="Total_Trans_Amt", y="Total_Trans_Ct",
                hue="Trang_Thai", palette=colors, alpha=0.65, ax=axes[1], s=35)
axes[1].set_title("Tương quan Giá trị Giao dịch vs. Số lần Giao dịch", fontweight="bold")
axes[1].set_xlabel("Tổng giá trị giao dịch (USD - Total_Trans_Amt)")
axes[1].set_ylabel("Tổng số lần giao dịch (Total_Trans_Ct)")

plt.tight_layout()
plt.savefig("charts/03_transaction_behavior.png")
plt.close()

# --- CHART 4: Inactivity & Contacts Count ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

inact_churn = df.groupby("Months_Inactive_12_mon")["Churn"].mean() * 100
sns.barplot(x=inact_churn.index, y=inact_churn.values, palette="YlOrRd", ax=axes[0], edgecolor="black")
for i, v in enumerate(inact_churn.values):
    axes[0].text(i, v + 1, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)
axes[0].set_title("Tỷ lệ Rời bỏ theo Số tháng Không hoạt động (12 tháng qua)", fontweight="bold")
axes[0].set_xlabel("Số tháng không giao dịch (Months_Inactive_12_mon)")
axes[0].set_ylabel("Tỷ lệ rời bỏ (%)")
axes[0].set_ylim(0, max(inact_churn.values) * 1.18)

cont_churn = df.groupby("Contacts_Count_12_mon")["Churn"].mean() * 100
sns.barplot(x=cont_churn.index, y=cont_churn.values, palette="PuBuGn", ax=axes[1], edgecolor="black")
for i, v in enumerate(cont_churn.values):
    axes[1].text(i, v + 1.5, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=9)
axes[1].set_title("Tỷ lệ Rời bỏ theo Số lần Liên hệ Ngân hàng (12 tháng qua)", fontweight="bold")
axes[1].set_xlabel("Số lần liên hệ với Ngân hàng (Contacts_Count_12_mon)")
axes[1].set_ylabel("Tỷ lệ rời bỏ (%)")
axes[1].set_ylim(0, 115)

plt.tight_layout()
plt.savefig("charts/04_inactivity_contacts.png")
plt.close()

# --- CHART 5: Correlation Heatmap ---
num_cols = [
    "Customer_Age", "Dependent_count", "Months_on_book", "Total_Relationship_Count",
    "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Credit_Limit", "Total_Revolving_Bal",
    "Total_Amt_Chng_Q4_Q1", "Total_Trans_Amt", "Total_Trans_Ct", "Total_Ct_Chng_Q4_Q1",
    "Avg_Utilization_Ratio", "Churn"
]
fig, ax = plt.subplots(figsize=(12, 9.5))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.6, vmax=1.0,
            linewidths=0.5, ax=ax, annot_kws={"size": 8.5})
ax.set_title("Ma trận Tương quan giữa các Biến Định lượng và Biến Mục tiêu (Churn)", fontsize=13, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig("charts/05_correlation_heatmap.png")
plt.close()

# --- 2. Data Preprocessing & Modeling ---
feature_cols = [c for c in df.columns if c not in ["Attrition_Flag", "Churn", "Trang_Thai"]]
X = df[feature_cols].copy()
y = df["Churn"].copy()

# Encode categorical variables
cat_cols = ["Gender", "Education_Level", "Marital_Status", "Income_Category", "Card_Category"]
X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=15, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1),
}

results = {}
roc_data = {}
cms = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "Accuracy": round(float(acc * 100), 2),
        "Precision": round(float(prec * 100), 2),
        "Recall": round(float(rec * 100), 2),
        "F1_Score": round(float(f1 * 100), 2),
        "ROC_AUC": round(float(auc * 100), 2),
        "Confusion_Matrix": cm.tolist(),
    }
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, auc)
    cms[name] = cm

summary["models"] = results
summary["train_size"] = int(len(X_train))
summary["test_size"] = int(len(X_test))

# Group stats for report insights
summary["group_means"] = df.groupby("Attrition_Flag")[
    ["Total_Trans_Ct", "Total_Trans_Amt", "Total_Revolving_Bal",
     "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio"]
].mean().round(2).to_dict()

# --- CHART 6: ROC Curve & Metrics Bar Chart ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

model_colors = {
    "Logistic Regression": "#1f77b4",
    "Decision Tree": "#ff7f0e",
    "Random Forest": "#2ca02c"
}
for name, (fpr, tpr, auc) in roc_data.items():
    axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc*100:.2f}%)", linewidth=2.3, color=model_colors[name])
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Ngẫu nhiên (AUC = 50%)")
axes[0].set_title("Đường cong ROC của 3 Thuật toán Phân loại", fontweight="bold")
axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
axes[0].set_ylabel("True Positive Rate (Recall)")
axes[0].legend(loc="lower right")

metrics_df = pd.DataFrame(results).T[["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]]
metrics_df.plot(kind="bar", ax=axes[1], width=0.78, edgecolor="black", colormap="Set2")
axes[1].set_title("So sánh Hiệu năng giữa các Thuật toán (%)", fontweight="bold")
axes[1].set_ylabel("Điểm số (%)")
axes[1].set_ylim(50, 105)
axes[1].set_xticklabels(metrics_df.index, rotation=0, fontweight="bold")
axes[1].legend(loc="lower right", ncol=2, fontsize=9)

plt.tight_layout()
plt.savefig("charts/06_model_comparison_roc.png")
plt.close()

# --- CHART 7: Confusion Matrices ---
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for idx, (name, cm) in enumerate(cms.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[idx],
                xticklabels=["Hiện hữu (0)", "Rời bỏ (1)"],
                yticklabels=["Hiện hữu (0)", "Rời bỏ (1)"],
                annot_kws={"size": 12, "weight": "bold"})
    axes[idx].set_title(f"{name}\n(Acc: {results[name]['Accuracy']}%, F1: {results[name]['F1_Score']}%)", fontweight="bold")
    axes[idx].set_xlabel("Dự đoán (Predicted)")
    axes[idx].set_ylabel("Thực tế (Actual)")
plt.tight_layout()
plt.savefig("charts/07_confusion_matrices.png")
plt.close()

# --- CHART 8: Feature Importance ---
rf_model = models["Random Forest"]
importances = pd.Series(rf_model.feature_importances_, index=X_encoded.columns).sort_values(ascending=False).head(12)
summary["top_features"] = importances.round(4).to_dict()

fig, ax = plt.subplots(figsize=(9, 5.5))
sns.barplot(x=importances.values * 100, y=importances.index, palette="viridis", ax=ax, edgecolor="black")
for i, v in enumerate(importances.values * 100):
    ax.text(v + 0.3, i, f"{v:.2f}%", va="center", fontweight="bold", fontsize=9.5)
ax.set_title("Top 12 Yếu tố Quan trọng nhất Dự đoán Rời bỏ Thẻ tín dụng (Random Forest)", fontweight="bold", pad=12)
ax.set_xlabel("Mức độ đóng góp vào mô hình (Feature Importance %)")
ax.set_xlim(0, max(importances.values * 100) * 1.15)
plt.tight_layout()
plt.savefig("charts/08_feature_importance.png")
plt.close()

with open("results_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("Analysis completed successfully! Summary saved to results_summary.json")
