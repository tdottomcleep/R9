# Example Datasets

This directory contains sample datasets for testing and demonstrating the AI Statistical Analysis App.

## Datasets

### 1. Sample Medical Data (`sample_medical_data.csv`)
- **Purpose**: General medical research demonstration
- **Size**: 20 patients
- **Variables**: 
  - Patient demographics (age, gender, weight, height)
  - Clinical measurements (blood pressure)
  - Treatment information
  - Outcome scores
  - Follow-up data

**Use Cases**:
- Descriptive statistics
- Treatment outcome analysis
- Correlation studies
- Basic hypothesis testing

### 2. Clinical Trial Data (`clinical_trial_data.csv`)
- **Purpose**: Randomized controlled trial analysis
- **Size**: 20 subjects
- **Variables**:
  - Randomization details
  - Treatment groups (Active vs Placebo)
  - Longitudinal measurements (baseline, week 4, 8, 12)
  - Safety data (adverse events, dropouts)
  - Compliance rates

**Use Cases**:
- Primary efficacy analysis
- Safety analysis
- Longitudinal data analysis
- Intent-to-treat analysis

### 3. Survival Data (`survival_data.csv`)
- **Purpose**: Survival analysis demonstration
- **Size**: 20 patients
- **Variables**:
  - Patient demographics
  - Cancer types and stages
  - Treatment modalities
  - Survival times and events
  - Risk factors

**Use Cases**:
- Kaplan-Meier survival curves
- Cox proportional hazards modeling
- Risk factor analysis
- Treatment comparison

## How to Use

### Upload Process
1. Start the AI Statistical Analysis App
2. Click "Upload CSV" in the left panel
3. Select one of the example datasets
4. Review the data preview
5. Begin your analysis

### Example Analysis Workflows

#### For Sample Medical Data
```python
# Exploratory data analysis
print("Dataset Overview:")
print(df.describe())
print("\nBlood pressure by treatment:")
print(df.groupby('treatment')['blood_pressure_systolic'].mean())

# Statistical test
from scipy import stats
hypertension_patients = df[df['diagnosis'] == 'Hypertension']
normal_patients = df[df['diagnosis'] == 'Normal']
t_stat, p_value = stats.ttest_ind(
    hypertension_patients['blood_pressure_systolic'],
    normal_patients['blood_pressure_systolic']
)
print(f"T-test result: t={t_stat:.3f}, p={p_value:.3f}")

# Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
df.boxplot(column='blood_pressure_systolic', by='diagnosis')
plt.title('Blood Pressure by Diagnosis')
plt.show()
```

#### For Clinical Trial Data
```python
# Primary efficacy analysis
print("Treatment Group Comparison:")
active_group = df[df['treatment_group'] == 'Active']
placebo_group = df[df['treatment_group'] == 'Placebo']

print(f"Active group week 12 score: {active_group['week_12_score'].mean():.2f}")
print(f"Placebo group week 12 score: {placebo_group['week_12_score'].mean():.2f}")

# Statistical comparison
from scipy import stats
t_stat, p_value = stats.ttest_ind(
    active_group['week_12_score'],
    placebo_group['week_12_score']
)
print(f"Primary efficacy p-value: {p_value:.3f}")

# Longitudinal analysis
import matplotlib.pyplot as plt
weeks = ['baseline_score', 'week_4_score', 'week_8_score', 'week_12_score']
active_means = [active_group[week].mean() for week in weeks]
placebo_means = [placebo_group[week].mean() for week in weeks]

plt.figure(figsize=(10, 6))
plt.plot([0, 4, 8, 12], active_means, 'o-', label='Active', linewidth=2)
plt.plot([0, 4, 8, 12], placebo_means, 's-', label='Placebo', linewidth=2)
plt.xlabel('Weeks')
plt.ylabel('Score')
plt.title('Treatment Effect Over Time')
plt.legend()
plt.grid(True)
plt.show()
```

#### For Survival Data
```python
# Kaplan-Meier survival analysis
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Overall survival
kmf = KaplanMeierFitter()
kmf.fit(df['survival_time_months'], df['event_occurred'])

plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Overall Survival Curve')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.grid(True)
plt.show()

print(f"Median survival time: {kmf.median_survival_time_:.1f} months")

# Survival by treatment
treatments = df['treatment'].unique()
plt.figure(figsize=(12, 6))

for treatment in treatments:
    treatment_data = df[df['treatment'] == treatment]
    kmf.fit(treatment_data['survival_time_months'], 
            treatment_data['event_occurred'], 
            label=treatment)
    kmf.plot_survival_function()

plt.title('Survival by Treatment')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.legend()
plt.grid(True)
plt.show()
```

## Data Quality Notes

- **Realistic but synthetic**: These datasets are created for demonstration purposes
- **Complete cases**: No missing values for simplicity
- **Appropriate distributions**: Values follow realistic ranges for medical data
- **Ethical considerations**: All data is synthetic and does not represent real patients

## Creating Your Own Test Data

### Guidelines
1. **Use realistic ranges** for medical variables
2. **Include appropriate missing values** for real-world scenarios
3. **Consider data distributions** (normal, skewed, etc.)
4. **Add relevant covariates** for your analysis type
5. **Include metadata** explaining variable definitions

### Python Code for Data Generation
```python
import pandas as pd
import numpy as np

# Generate synthetic medical data
np.random.seed(42)
n_patients = 100

data = {
    'patient_id': [f'P{i:03d}' for i in range(1, n_patients + 1)],
    'age': np.random.normal(55, 15, n_patients).astype(int),
    'gender': np.random.choice(['M', 'F'], n_patients),
    'weight': np.random.normal(70, 12, n_patients),
    'height': np.random.normal(170, 8, n_patients),
    # Add more variables as needed
}

df = pd.DataFrame(data)
df.to_csv('your_synthetic_data.csv', index=False)
```

## Best Practices

1. **Start with exploratory analysis** to understand your data
2. **Use appropriate statistical tests** for your data type
3. **Visualize your results** to communicate findings
4. **Document your analysis steps** for reproducibility
5. **Consider clinical significance** alongside statistical significance

---

**Happy analyzing with the example datasets! 📊🔬**