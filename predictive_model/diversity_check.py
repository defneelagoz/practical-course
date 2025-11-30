import pandas as pd

# Load dataset
df = pd.read_csv("/Users/aybikealtunbas/practical-course/predictive_model/student_data.csv", sep=";")

# -------------------- 1. Numeric summary --------------------
print("=== Numeric Summary ===")
print(df.describe())

# -------------------- 2. Categorical summary --------------------
cat_cols = df.select_dtypes(include=["object", "category"]).columns
for col in cat_cols:
    print(f"\n=== Value counts for {col} ===")
    print(df[col].value_counts())

# -------------------- 3. Class balance --------------------
if "Target" in df.columns:
    print("\n=== Target distribution ===")
    print(df["Target"].value_counts(normalize=True))

# -------------------- 4. Rare categories --------------------
threshold = 0.05  # less than 5% of dataset
for col in cat_cols:
    freq = df[col].value_counts(normalize=True)
    rare = freq[freq < threshold]
    if not rare.empty:
        print(f"\nRare categories in {col}:")
        print(rare)

# -------------------- 5. Cross-tab example --------------------
# Example: nationality × scholarship × Target
for col1 in ["Nationality", "Scholarship_Status"]:
    if col1 in df.columns:
        print(f"\nCross-tab: {col1} × Target")
        print(pd.crosstab(df[col1], df["Target"], margins=True))
