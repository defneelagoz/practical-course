import pandas as pd
import scipy.stats as stats

# Load the dataframe. The snippet indicates ';' delimiter and potential comma for decimals in scientific notation.
df = pd.read_csv('student_data_2.csv', sep=';', decimal=',')

# Inspect the dataframe
print(df.head())
print(df.info())
print(df['Output'].unique())

import numpy as np

# Function to clean grade columns
def clean_grade(x):
    if isinstance(x, str):
        # Replace comma with dot
        x = x.replace(',', '.')
        try:
            val = float(x)
        except:
            return np.nan
    else:
        val = x
    
    # Heuristic to fix data issues:
    # If val is huge (e.g. > 100), it's likely an error or Excel date. 
    # Valid grades are typically 0-20.
    if val > 20:
        return np.nan # Treat as missing for now to avoid skewing normality check
    return val

# Apply cleaning
df['Curricular units 1st sem (grade)'] = df['Curricular units 1st sem (grade)'].apply(clean_grade)
df['Curricular units 2nd sem (grade)'] = df['Curricular units 2nd sem (grade)'].apply(clean_grade)

# Check info after cleaning
print(df[['Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']].describe())
print(df.info())

import matplotlib.pyplot as plt
import seaborn as sns

variables = [
    'Age at enrollment', 
    'Curricular units 1st sem (enrolled)', 
    'Curricular units 1st sem (approved)', 
    'Curricular units 1st sem (grade)', 
    'Curricular units 2nd sem (enrolled)', 
    'Curricular units 2nd sem (approved)', 
    'Curricular units 2nd sem (grade)'
]

categories = ['Dropout', 'Graduate', 'Enrolled']
results = []

for var in variables:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Distribution of {var} by Output Category', fontsize=16)
    
    for i, cat in enumerate(categories):
        subset = df[df['Output'] == cat][var].dropna()
        ax = axes[i]
        
        # Plot
        sns.histplot(subset, kde=True, ax=ax, color='skyblue')
        
        # Normality Test
        if len(subset) >= 3: # Need at least 3 data points for Shapiro
            stat, p_value = stats.shapiro(subset)
            normal_status = "Normal" if p_value > 0.05 else "Not Normal"
        else:
            p_value = np.nan
            normal_status = "N/A"
            
        ax.set_title(f'{cat}\nShapiro p={p_value:.2e}\n({normal_status})')
        ax.set_xlabel(var)
        
        results.append({
            'Variable': var,
            'Category': cat,
            'P-Value': p_value,
            'Is Normal (p>0.05)': normal_status
        })
        
    plt.tight_layout()
    plt.savefig(f'{var}_distribution.png')

# Create summary dataframe
results_df = pd.DataFrame(results)
print(results_df)
results_df.to_csv('normality_test_results.csv', index=False)