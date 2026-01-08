# eda.py - Exploratory Data Analysis for Paisabazar dataset
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = 'data/paisabazar_sample.csv'
OUT_DIR = 'eda_outputs'
os.makedirs(OUT_DIR, exist_ok=True)

def main():
    df = pd.read_csv(DATA_PATH)
    print('Shape:', df.shape)
    print('\nColumns:', df.columns.tolist())
    print('\nHead:\n', df.head().to_string(index=False))
    print('\nDescribe:\n', df.describe(include='all').T)

    # Missing values
    mv = df.isnull().sum()
    print('\nMissing values:\n', mv[mv>0])

    # Categorical distributions
    cat_cols = ['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
    for c in cat_cols:
        plt.figure(figsize=(6,4))
        sns.countplot(data=df, x=c, order=df[c].value_counts().index)
        plt.title(f'Distribution of {c}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{c}_dist.png'))
        plt.close()

    # Numerical distributions
    num_cols = ['ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term']
    for c in num_cols:
        plt.figure(figsize=(6,4))
        sns.histplot(df[c], kde=True)
        plt.title(c)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{c}_hist.png'))
        plt.close()

    # Relationship between income and loan amount
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Loan_Status')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'income_vs_loan.png'))
    plt.close()

    print('EDA plots saved to', OUT_DIR)

if __name__ == '__main__':
    main()
