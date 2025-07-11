# Usage Guide

## Getting Started

### 1. First Steps
1. **Start the application** following the [Installation Guide](INSTALLATION.md)
2. **Open your browser** to http://localhost:3000
3. **Prepare your CSV data** for analysis

### 2. Interface Overview

#### 3-Panel Layout
- **Left Panel**: Session management and file upload
- **Center Panel**: Chat interface with AI assistant
- **Right Panel**: Results, history, and data preview

#### Key Features
- **Collapsible panels** for customized workspace
- **Real-time chat** with Gemini AI
- **Integrated code execution** with immediate results
- **Session persistence** across browser sessions

## Basic Workflow

### Step 1: Upload Your Data
1. Click the **"Upload CSV"** button in the left panel
2. Select your CSV file (medical/healthcare data recommended)
3. Wait for file validation and preview generation
4. Review the **data preview** in the right panel

### Step 2: Start Analysis
1. Use the **chat interface** to ask questions about your data
2. Get **AI-powered suggestions** for appropriate analyses
3. **Execute Python code** directly in the sandbox
4. View **results and visualizations** in real-time

### Step 3: Explore Results
1. Switch between **Execution** and **History** tabs
2. Review **sectioned analysis** results
3. Access **structured analysis** summaries
4. Export or save important findings

## Advanced Features

### Julius AI-Style Sectioned Execution

#### What It Does
- Automatically organizes your code into logical sections
- Classifies sections by type (summary, statistical tests, visualizations)
- Provides structured metadata for each section
- Enables better analysis organization and reproducibility

#### How to Use
1. Write your analysis code in the chat interface
2. The system automatically detects and sections your code
3. View organized results with clear section headers
4. Access metadata and context for each section

#### Example
```python
# This code will be automatically sectioned
# Data Overview
print('Dataset shape:', df.shape)
print('Column names:', df.columns.tolist())

# Descriptive Statistics
print(df.describe())

# Statistical Test
from scipy import stats
result = stats.ttest_ind(df[df['gender']=='M']['age'], df[df['gender']=='F']['age'])
print(f'T-test result: t={result.statistic:.3f}, p={result.pvalue:.3f}')

# Visualization
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
df.boxplot(column='age', by='gender')
plt.title('Age Distribution by Gender')
plt.show()
```

### Healthcare-Specific Analysis Types

#### Clinical Trial Analysis
```python
# Analyze treatment efficacy
treatment_group = df[df['treatment'] == 'active']
control_group = df[df['treatment'] == 'placebo']

# Efficacy analysis
efficacy_result = stats.ttest_ind(treatment_group['outcome'], control_group['outcome'])
print(f'Treatment efficacy: p={efficacy_result.pvalue:.3f}')

# Safety analysis
adverse_events = df.groupby('treatment')['adverse_events'].sum()
print('Adverse events by treatment:', adverse_events)
```

#### Survival Analysis
```python
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Kaplan-Meier survival analysis
kmf = KaplanMeierFitter()
kmf.fit(df['survival_time'], df['event_observed'])

# Plot survival curve
plt.figure(figsize=(10, 6))
kmf.plot_survival_function()
plt.title('Kaplan-Meier Survival Curve')
plt.xlabel('Time (months)')
plt.ylabel('Survival Probability')
plt.show()

# Print median survival time
print(f'Median survival time: {kmf.median_survival_time_:.2f} months')
```

#### Epidemiological Analysis
```python
# Calculate incidence rates
person_years = df['follow_up_time'].sum()
events = df['event_occurred'].sum()
incidence_rate = events / person_years * 1000

print(f'Incidence rate: {incidence_rate:.2f} per 1000 person-years')

# Age-standardized rates
age_groups = df.groupby('age_group')
standardized_rates = age_groups.apply(lambda x: (x['events'].sum() / x['person_time'].sum()) * 1000)
print('Age-standardized incidence rates:', standardized_rates)
```

## Common Use Cases

### 1. Medical Research Data Analysis

#### Dataset Preparation
```python
# Load and explore medical dataset
print('Dataset overview:')
print(f'Shape: {df.shape}')
print(f'Columns: {df.columns.tolist()}')
print(f'Missing values: {df.isnull().sum()}')

# Data quality assessment
print('\nData quality metrics:')
print(f'Duplicate records: {df.duplicated().sum()}')
print(f'Complete cases: {df.dropna().shape[0]}')
```

#### Demographic Analysis
```python
# Patient demographics
print('Patient Demographics:')
print(f'Age: {df["age"].mean():.1f} ± {df["age"].std():.1f} years')
print(f'Gender distribution: {df["gender"].value_counts()}')
print(f'Diagnosis distribution: {df["diagnosis"].value_counts()}')

# Visualization
import seaborn as sns
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
df['age'].hist(bins=20)
plt.title('Age Distribution')

plt.subplot(1, 3, 2)
df['gender'].value_counts().plot(kind='bar')
plt.title('Gender Distribution')

plt.subplot(1, 3, 3)
df['diagnosis'].value_counts().plot(kind='pie')
plt.title('Diagnosis Distribution')

plt.tight_layout()
plt.show()
```

### 2. Clinical Trial Analysis

#### Primary Outcome Analysis
```python
# Primary efficacy analysis
primary_outcome = 'blood_pressure_reduction'
treatment_groups = df['treatment_group'].unique()

print('Primary Outcome Analysis:')
for group in treatment_groups:
    group_data = df[df['treatment_group'] == group][primary_outcome]
    print(f'{group}: {group_data.mean():.2f} ± {group_data.std():.2f}')

# Statistical comparison
from scipy.stats import f_oneway
groups = [df[df['treatment_group'] == group][primary_outcome] for group in treatment_groups]
f_stat, p_value = f_oneway(*groups)
print(f'\nANOVA result: F={f_stat:.3f}, p={p_value:.3f}')
```

#### Safety Analysis
```python
# Adverse event analysis
ae_summary = df.groupby('treatment_group')['adverse_events'].agg(['count', 'sum', 'mean'])
print('Adverse Events Summary:')
print(ae_summary)

# Serious adverse events
serious_ae = df[df['serious_ae'] == True]
print(f'\nSerious adverse events: {len(serious_ae)} ({len(serious_ae)/len(df)*100:.1f}%)')
```

### 3. Biostatistical Analysis

#### Correlation Analysis
```python
# Correlation matrix for continuous variables
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = df[numeric_cols].corr()

# Visualization
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Significant correlations
significant_correlations = correlation_matrix[abs(correlation_matrix) > 0.5]
print('Significant correlations (>0.5):')
print(significant_correlations)
```

#### Regression Analysis
```python
# Multiple regression analysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Prepare data
X = df[['age', 'weight', 'height']]
y = df['blood_pressure']

# Fit model
model = LinearRegression()
model.fit(X, y)

# Results
print('Regression Results:')
print(f'R-squared: {model.score(X, y):.3f}')
print('Coefficients:')
for feature, coef in zip(X.columns, model.coef_):
    print(f'  {feature}: {coef:.3f}')
print(f'Intercept: {model.intercept_:.3f}')
```

## Tips and Best Practices

### 1. Data Quality
- **Always start with data exploration** (`df.info()`, `df.describe()`)
- **Check for missing values** and handle appropriately
- **Validate data types** and convert if necessary
- **Look for outliers** and investigate anomalies

### 2. Statistical Analysis
- **Choose appropriate tests** based on data type and distribution
- **Check assumptions** before applying statistical tests
- **Use appropriate effect size measures** alongside p-values
- **Apply multiple comparison corrections** when needed

### 3. Visualization
- **Use appropriate chart types** for your data
- **Include clear titles and labels**
- **Consider color accessibility** in your plots
- **Export high-quality images** for reports

### 4. Code Organization
- **Use comments** to explain your analysis steps
- **Break complex analysis into sections**
- **Use descriptive variable names**
- **Document your assumptions and decisions**

### 5. Result Interpretation
- **Provide context** for your findings
- **Discuss clinical significance** alongside statistical significance
- **Acknowledge limitations** of your analysis
- **Suggest next steps** for further investigation

## Troubleshooting

### Common Issues

#### Data Upload Problems
- **File format**: Ensure your file is CSV format
- **File size**: Maximum 100MB per file
- **Encoding**: Use UTF-8 encoding for special characters
- **Column names**: Avoid spaces and special characters

#### Code Execution Errors
- **Syntax errors**: Check Python syntax carefully
- **Variable names**: Use `df` to reference your uploaded data
- **Library imports**: Common libraries are pre-imported
- **Memory limits**: Large datasets may require optimization

#### API Key Issues
- **Invalid key**: Verify your Gemini API key format
- **Rate limits**: Wait if you hit rate limits
- **Permissions**: Ensure API key has necessary permissions
- **Billing**: Check if billing is enabled for your account

### Getting Help

1. **Check the error message** for specific guidance
2. **Review the API documentation** for endpoint details
3. **Use the chat interface** to ask for help
4. **Check the troubleshooting section** in the installation guide
5. **Create an issue** if you find a bug

## Next Steps

### Advanced Features to Explore
- **Custom analysis templates** for common workflows
- **Automated report generation** for standardized outputs
- **Integration with external databases** for larger datasets
- **Batch processing** for multiple files

### Learning Resources
- **Statistical analysis guides** for healthcare data
- **Python data science tutorials** for advanced techniques
- **Medical research methodology** for proper study design
- **Data visualization best practices** for clear communication

---

**Happy analyzing! 🔬📊**