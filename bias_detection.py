import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import pickle 

# Define and load dataset
file_path = r"C:\Users\divy\AI\finalProject\compas-scores-raw.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"File not found: {file_path}")

# Drop unnecessary columns
columns_to_drop = [
    'Person_ID', 'AssessmentID', 'Case_ID', 'Agency_Text', 'Ethnic_Code_Text',
    'FirstName', 'LastName', 'MiddleName', 'DateOfBirth', 'ScaleSet_ID',
    'RawScore', 'DisplayText', 'Screening_Date', 'IsDeleted'
]
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Handle missing data - drop rows with any missing values
df.dropna(inplace=True)

# Encode categorical variables
# Encode `ScoreText` (Low, Medium, High) into numeric values
df['ScoreText'] = df['ScoreText'].map({'Low': 0, 'Medium': 1, 'High': 2}).fillna(0).astype(int)

# Encode `Language` and `MaritalStatus` as categorical codes
df['Language'] = pd.Categorical(df['Language']).codes
df['MaritalStatus'] = pd.Categorical(df['MaritalStatus']).codes

# Normalize numerical variables
scaler = MinMaxScaler()
if 'DecileScore' in df.columns:
    df['DecileScore'] = scaler.fit_transform(df[['DecileScore']])

# Simulate 'PredictedRisk' column if missing
if 'PredictedRisk' not in df.columns:
    df['PredictedRisk'] = np.random.choice([0, 1], size=len(df))  # Simulated binary predictions

# Save preprocessed data for future use
output_path = r"C:\Users\divy\AI\finalProject\preprocessed_compas.csv"
df.to_csv(output_path, index=False)

# PHASE 1: Implementing Different Fairness Metrics
# 1. Demographic Parity
def calculate_demographic_parity_all_columns(data, outcome_column):
    relevant_columns = ['Language', 'MaritalStatus', 'Sex_Code_Text', 
                        'RecSupervisionLevel']
    all_decision_rates = {}
    for column in relevant_columns:
        if column in data.columns:
            decision_rates = {}
            for group in data[column].unique():
                group_data = data[data[column] == group]
                decision_rate = group_data[outcome_column].mean()  # Proportion of positive outcomes
                decision_rates[group] = decision_rate
            all_decision_rates[column] = decision_rates
    return all_decision_rates

# Calculate decision rates
outcome_column = 'ScoreText'
decision_rates = calculate_demographic_parity_all_columns(df, outcome_column)

# Plot bar charts for each relevant column
def plot_decision_rate_bar_charts(decision_rates, save_dir=None):
    for column, rates in decision_rates.items():
        plt.figure(figsize=(10, 6))
        groups = list(rates.keys())
        values = list(rates.values())
        plt.bar(groups, values, alpha=0.7)
        plt.title(f'Decision Rates by {column}', fontsize=14)
        plt.xlabel(f'{column} Groups', fontsize=12)
        plt.ylabel('Decision Rate', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_name = f'{column}_decision_rates.png'
            plt.savefig(os.path.join(save_dir, file_name))
        else:
            plt.show()

# Define directory to save plots
save_directory = r"C:\Users\divy\AI\finalProject\demoParitycharts"
plot_decision_rate_bar_charts(decision_rates, save_dir=save_directory)

# 2. Equalized Odds
def calculate_equalized_odds_all_columns(data, outcome_column, prediction_column):
    """
    Calculate True Positive Rate (TPR) and False Positive Rate (FPR) for all relevant demographic columns.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - outcome_column: Column representing the true outcome (e.g., 'ScoreText').
    - prediction_column: Column representing the predicted outcome (e.g., 'PredictedRisk').

    Returns:
    - A dictionary where keys are column names and values are dictionaries of TPR and FPR for each group.
    """
    relevant_columns = ['Language', 'MaritalStatus', 'Sex_Code_Text', 'RecSupervisionLevel']
    all_equalized_odds = {}

    for column in relevant_columns:
        if column in data.columns:
            equalized_odds = {}
            for group in data[column].unique():
                group_data = data[data[column] == group]
                y_true = group_data[outcome_column]
                y_pred = group_data[prediction_column]

                # Compute confusion matrix components
                try:
                    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                except ValueError:
                    # Handle cases where confusion_matrix fails (e.g., all values in y_true are 0 or 1)
                    tpr = fpr = 0

                equalized_odds[group] = {'TPR': tpr, 'FPR': fpr}

            all_equalized_odds[column] = equalized_odds

    return all_equalized_odds


def plot_equalized_odds_all_columns(all_equalized_odds, save_dir=None):
    """
    Plot grouped bar charts for TPR and FPR for each group in all demographic columns.

    Parameters:
    - all_equalized_odds: Dictionary containing TPR and FPR for each group in each column.
    - save_dir: Directory to save the plots. If None, the plots are displayed instead.
    """
    for column, odds in all_equalized_odds.items():
        groups = list(odds.keys())
        tpr_values = [odds[group]['TPR'] for group in groups]
        fpr_values = [odds[group]['FPR'] for group in groups]

        x = range(len(groups))  # Positions for the groups
        bar_width = 0.4  # Width of each bar

        plt.figure(figsize=(12, 6))

        # Plot TPR and FPR as grouped bars
        plt.bar(x, tpr_values, width=bar_width, label='True Positive Rate (TPR)', alpha=0.7)
        plt.bar([i + bar_width for i in x], fpr_values, width=bar_width, label='False Positive Rate (FPR)', alpha=0.7)

        # Formatting the plot
        plt.title(f'Equalized Odds by {column}', fontsize=14)
        plt.xlabel(f'{column} Groups', fontsize=12)
        plt.ylabel('Rate', fontsize=12)
        plt.xticks([i + bar_width / 2 for i in x], groups, rotation=45, ha='right')
        plt.legend(fontsize=12)
        plt.tight_layout()

        # Save or show the plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_name = f'{column}_equalized_odds.png'
            plt.savefig(os.path.join(save_dir, file_name))
        else:
            plt.show()


# Example usage
outcome_column = 'ScoreText'
prediction_column = 'PredictedRisk'

# Calculate Equalized Odds for all relevant columns
all_equalized_odds = calculate_equalized_odds_all_columns(df, outcome_column, prediction_column)

# Plot and save the charts for all columns
plot_save_directory = r"C:\Users\divy\AI\finalProject\eqOddsCharts"
plot_equalized_odds_all_columns(all_equalized_odds, save_dir=plot_save_directory)

# 3. Disparate Impact
def calculate_disparate_impact_all_columns(data, outcome_column, favorable_outcome=1):
    """
    Calculate Disparate Impact for all relevant demographic columns.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - outcome_column: Column representing the outcome of interest (e.g., 'HighRisk').
    - favorable_outcome: The value representing a favorable outcome (e.g., 1 for "low risk").

    Returns:
    - A dictionary with decision rates and Disparate Impact ratios for each column.
    """
    relevant_columns = ['Language', 'MaritalStatus', 'Sex_Code_Text', 'RecSupervisionLevel']
    all_disparate_impact = {}

    for column in relevant_columns:
        if column in data.columns:
            decision_rates = {}
            for group in data[column].unique():
                group_data = data[data[column] == group]
                favorable_rate = (group_data[outcome_column] == favorable_outcome).mean()
                decision_rates[group] = favorable_rate

            # Calculate Disparate Impact (ratio of the lowest to the highest decision rate)
            max_rate = max(decision_rates.values())
            min_rate = min(decision_rates.values())
            disparate_impact_ratio = min_rate / max_rate if max_rate > 0 else 0

            all_disparate_impact[column] = {
                'decision_rates': decision_rates,
                'disparate_impact_ratio': disparate_impact_ratio
            }

    return all_disparate_impact


def plot_disparate_impact_all_columns(all_disparate_impact, save_dir=None):
    """
    Plot Disparate Impact decision rates for all relevant demographic columns.

    Parameters:
    - all_disparate_impact: Dictionary with decision rates and Disparate Impact ratios for each column.
    - save_dir: Directory to save the plots. If None, the plots are displayed instead.
    """
    for column, metrics in all_disparate_impact.items():
        decision_rates = metrics['decision_rates']
        disparate_impact_ratio = metrics['disparate_impact_ratio']

        groups = list(decision_rates.keys())
        rates = list(decision_rates.values())

        plt.figure(figsize=(12, 6))
        plt.bar(groups, rates, alpha=0.7)
        plt.title(f'Disparate Impact by {column} (Ratio: {disparate_impact_ratio:.2f})', fontsize=14)
        plt.xlabel(f'{column} Groups', fontsize=12)
        plt.ylabel('Favorable Outcome Rate', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Save or show plot
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            file_name = f'{column}_disparate_impact.png'
            plt.savefig(os.path.join(save_dir, file_name))
        else:
            plt.show()


# Example usage
outcome_column = 'ScoreText'  # Replace with your outcome column
favorable_outcome = 1  # Define what constitutes a favorable outcome (e.g., 1 for "Low Risk")

# Calculate Disparate Impact for all relevant columns
all_disparate_impact = calculate_disparate_impact_all_columns(df, outcome_column, favorable_outcome)

# Plot and save the charts
plot_save_directory = r"C:\Users\divy\AI\finalProject\disparateImpactCharts"
plot_disparate_impact_all_columns(all_disparate_impact, save_dir=plot_save_directory)


# PHASE 2: Calculate Statistical Parity Difference
def calculate_statistical_parity_difference(data, outcome_column, favorable_outcome=1):
    """
    Calculate Statistical Parity Difference (SPD) for relevant demographic columns.

    Parameters:
    - data: Pandas DataFrame containing the dataset.
    - outcome_column: Column representing the outcome of interest (e.g., 'HighRisk').
    - favorable_outcome: The value representing a favorable outcome (e.g., 1 for "low risk").

    Returns:
    - A dictionary with SPD values for each relevant column.
    """
    relevant_columns = ['Language', 'MaritalStatus', 'Sex_Code_Text', 'RecSupervisionLevel']
    spd_results = {}

    for column in relevant_columns:
        if column in data.columns:
            group_rates = {}
            for group in data[column].unique():
                group_data = data[data[column] == group]
                positive_rate = (group_data[outcome_column] == favorable_outcome).mean()
                group_rates[group] = positive_rate

            # Compute SPD
            max_rate = max(group_rates.values())
            min_rate = min(group_rates.values())
            spd_results[column] = max_rate - min_rate

    return spd_results


# Example usage
outcome_column = 'ScoreText'  # Define your favorable outcome column
favorable_outcome = 1  # Example: 1 for "Low Risk"
spd_results = calculate_statistical_parity_difference(df, outcome_column, favorable_outcome)

# Print SPD results
print("Statistical Parity Difference Results:", spd_results)

def plot_statistical_parity_difference(spd_results, save_dir=None):
    """
    Plot Statistical Parity Difference for all relevant columns.

    Parameters:
    - spd_results: Dictionary with SPD values for each column.
    - save_dir: Directory to save the plots. If None, the plots are displayed instead.
    """
    columns = list(spd_results.keys())
    spd_values = list(spd_results.values())

    plt.figure(figsize=(10, 6))
    plt.bar(columns, spd_values, alpha=0.7)
    plt.title('Statistical Parity Difference by Column', fontsize=14)
    plt.xlabel('Demographic Columns', fontsize=12)
    plt.ylabel('SPD Value', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        file_name = 'statistical_parity_difference.png'
        plt.savefig(os.path.join(save_dir, file_name))
    else:
        plt.show()


# Example usage
plot_save_directory = r"C:\Users\divy\AI\finalProject\spdCharts"
plot_statistical_parity_difference(spd_results, save_dir=plot_save_directory)

# PHASE 3: Validation of Metrics (test on mock data)
# Mock Data for Validation
mock_data = pd.DataFrame({
    'Language': [0, 0, 1, 1],
    'MaritalStatus': [0, 1, 0, 1],
    'Sex_Code_Text': [0, 0, 1, 1],
    'RecSupervisionLevel': [1, 1, 0, 0],
    'ScoreText': [1, 0, 1, 0],  # True outcomes
    'PredictedRisk': [1, 0, 1, 0]  # Predicted outcomes
})

# Validate Demographic Parity
dp_results = calculate_demographic_parity_all_columns(mock_data, 'ScoreText')
print("Demographic Parity (Validation):", dp_results)

# Validate Equalized Odds
eo_results = calculate_equalized_odds_all_columns(mock_data, 'ScoreText', 'PredictedRisk')
print("Equalized Odds (Validation):", eo_results)

# Validate SPD
spd_results = calculate_statistical_parity_difference(mock_data, 'ScoreText', favorable_outcome=1)
print("Statistical Parity Difference (Validation):", spd_results)

# Save validation results to a text file
with open(r"C:\Users\divy\AI\finalProject\validation_results.txt", "w") as f:
    f.write(f"Statistical Parity Difference Results: {spd_results}\n")
    f.write(f"Demographic Parity (Validation): {dp_results}\n")
    f.write(f"Equalized Odds (Validation): {eo_results}\n")

# Combine all metric results into one dictionary
all_metrics_results = {
    'Statistical Parity Difference': spd_results,
    'Demographic Parity': dp_results,
    'Equalized Odds': eo_results
}

# Save the results to a .pkl file
metrics_file_path = r"C:\Users\divy\AI\finalProject\metrics_results.pkl"
with open(metrics_file_path, 'wb') as f:
    pickle.dump(all_metrics_results, f)

print(f"All metrics saved to {metrics_file_path}")
