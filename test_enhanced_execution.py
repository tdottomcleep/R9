#!/usr/bin/env python3
"""
Test enhanced code execution separately
"""

import requests
import json

# Create session with medical data
csv_data = '''patient_id,age,gender,blood_pressure_systolic,blood_pressure_diastolic,cholesterol,bmi,diabetes,heart_disease
P001,45,M,140,90,220,25.5,0,0
P002,32,F,120,80,200,22.1,0,0
P003,67,M,160,100,280,30.2,1,1
P004,55,F,135,85,240,27.8,1,0
P005,41,M,145,95,260,29.1,0,1'''

files = {'file': ('medical_data.csv', csv_data, 'text/csv')}
session_response = requests.post('https://0ee84439-001a-41bd-9cd3-14d4582c1370.preview.emergentagent.com/api/sessions', files=files)
session_id = session_response.json()['id']

# Test comprehensive analysis with proper string handling
test_code = '''
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from lifelines import KaplanMeierFitter

print("=== COMPREHENSIVE MEDICAL DATA ANALYSIS ===")
print("Dataset shape:", df.shape)

# 1. Basic stats
print("1. DESCRIPTIVE STATISTICS:")
print(df.describe())

# 2. T-test
print("3. STATISTICAL TESTS:")
male_bp = df[df['gender'] == 'M']['blood_pressure_systolic']
female_bp = df[df['gender'] == 'F']['blood_pressure_systolic']
if len(male_bp) > 0 and len(female_bp) > 0:
    t_stat, p_value = stats.ttest_ind(male_bp, female_bp)
    print(f"T-test (BP by gender): t={t_stat:.3f}, p={p_value:.3f}")

# 3. Chi-square test
chi2, p_chi2, dof, expected = stats.chi2_contingency(pd.crosstab(df['diabetes'], df['heart_disease']))
print(f"Chi-square (diabetes vs heart disease): χ²={chi2:.3f}, p={p_chi2:.3f}")

# 4. Logistic Regression
print("4. LOGISTIC REGRESSION (Statsmodels):")
X = df[['age', 'bmi', 'blood_pressure_systolic']]
y = df['heart_disease']
X = sm.add_constant(X)
logit_model = sm.Logit(y, X)
result = logit_model.fit(disp=0)
print(f"AIC: {result.aic:.2f}, Pseudo R²: {result.prsquared:.3f}")

# 5. Survival Analysis
print("5. SURVIVAL ANALYSIS SIMULATION:")
np.random.seed(42)
survival_times = np.random.exponential(10, len(df))
events = np.random.binomial(1, 0.7, len(df))
kmf = KaplanMeierFitter()
kmf.fit(survival_times, events)
print(f"Median survival time: {kmf.median_survival_time_:.2f} years")

print("✅ COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY")
print("All advanced statistical libraries working properly!")
'''

data = {
    'session_id': session_id,
    'code': test_code,
    'gemini_api_key': 'test_key'
}

response = requests.post(f'https://0ee84439-001a-41bd-9cd3-14d4582c1370.preview.emergentagent.com/api/sessions/{session_id}/execute', 
                        json=data, 
                        headers={'Content-Type': 'application/json'})

print(f'Status: {response.status_code}')
result = response.json()
print(f'Success: {result.get("success")}')
output = result.get('output', '')

# Check for key components
components = ['COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY', 'T-test', 'Chi-square', 'LOGISTIC REGRESSION', 'SURVIVAL ANALYSIS']
found_components = [comp for comp in components if comp in output]
print(f'Found components: {len(found_components)}/{len(components)}')

if result.get('success') and len(found_components) >= 4:
    print("✅ Enhanced code execution working - all advanced libraries functional")
else:
    print("❌ Enhanced code execution failed")
    if result.get('error'):
        print(f'Error: {result.get("error")}')
    print(f'Output: {output[:500]}...')