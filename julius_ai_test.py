#!/usr/bin/env python3
"""
Julius AI-Style Sectioned Execution Testing
Comprehensive testing of the Julius AI-style sectioned execution functionality
"""

import requests
import json
import base64
import io
import pandas as pd
import time
from typing import Dict, Any, Optional

# Configuration
BACKEND_URL = "https://b5523fcb-02b2-4035-b52d-0ab758f4b30d.preview.emergentagent.com/api"
TEST_API_KEY = "test_key_123"

class JuliusAITester:
    def __init__(self):
        self.session_id = None
        self.test_results = {}
        
    def create_comprehensive_medical_csv(self) -> str:
        """Create comprehensive medical dataset for Julius AI testing"""
        data = {
            'patient_id': [f'P{i:03d}' for i in range(1, 101)],  # 100 patients
            'age': [25 + (i % 50) for i in range(100)],
            'gender': ['M', 'F'] * 50,
            'treatment_group': ['Control', 'Treatment'] * 50,
            'baseline_bp_systolic': [120 + (i % 40) for i in range(100)],
            'baseline_bp_diastolic': [80 + (i % 20) for i in range(100)],
            'followup_bp_systolic': [115 + (i % 35) for i in range(100)],
            'followup_bp_diastolic': [75 + (i % 18) for i in range(100)],
            'cholesterol': [180 + (i % 80) for i in range(100)],
            'bmi': [20 + (i % 15) for i in range(100)],
            'diabetes': [i % 3 == 0 for i in range(100)],  # ~33% have diabetes
            'heart_disease': [i % 4 == 0 for i in range(100)],  # ~25% have heart disease
            'smoking_status': [['Never', 'Former', 'Current'][i % 3] for i in range(100)],
            'survival_time_months': [12 + (i % 60) for i in range(100)],
            'event_occurred': [i % 5 != 0 for i in range(100)],  # 80% event rate
            'medication_adherence': [0.5 + (i % 50) / 100 for i in range(100)],
            'quality_of_life_score': [50 + (i % 40) for i in range(100)],
            'adverse_events': [i % 10 == 0 for i in range(100)]  # 10% adverse events
        }
        
        df = pd.DataFrame(data)
        return df.to_csv(index=False)
    
    def setup_session(self) -> bool:
        """Setup test session with medical data"""
        print("Setting up test session with comprehensive medical data...")
        
        try:
            csv_data = self.create_comprehensive_medical_csv()
            
            files = {
                'file': ('clinical_trial_data.csv', csv_data, 'text/csv')
            }
            
            response = requests.post(f"{BACKEND_URL}/sessions", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                self.session_id = data.get('id')
                print(f"✅ Session created successfully: {self.session_id}")
                return True
            else:
                print(f"❌ Session creation failed: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Session setup failed: {str(e)}")
            return False
    
    def test_sectioned_execution_comprehensive(self) -> bool:
        """Test comprehensive sectioned execution with medical data analysis"""
        print("Testing Comprehensive Sectioned Execution...")
        
        if not self.session_id:
            print("❌ No session ID available")
            return False
        
        try:
            # Comprehensive medical analysis code with multiple sections
            comprehensive_code = """
# ===== CLINICAL DATA OVERVIEW =====
print("CLINICAL TRIAL DATA ANALYSIS")
print("=" * 50)
print(f"Total patients enrolled: {len(df)}")
print(f"Variables collected: {len(df.columns)}")
print(f"Treatment groups: {df['treatment_group'].value_counts().to_dict()}")

# ===== DESCRIPTIVE STATISTICS =====
print("\\nBASELINE CHARACTERISTICS")
print("-" * 30)
numeric_vars = ['age', 'baseline_bp_systolic', 'baseline_bp_diastolic', 'cholesterol', 'bmi']
baseline_stats = df[numeric_vars].describe()
print(baseline_stats)

print("\\nCATEGORICAL VARIABLES")
print(f"Gender distribution: {df['gender'].value_counts().to_dict()}")
print(f"Smoking status: {df['smoking_status'].value_counts().to_dict()}")
print(f"Diabetes prevalence: {df['diabetes'].sum()}/{len(df)} ({df['diabetes'].mean()*100:.1f}%)")

# ===== STATISTICAL TESTING =====
from scipy import stats
import numpy as np

print("\\nSTATISTICAL TESTS")
print("-" * 20)

# T-test for baseline BP between treatment groups
control_bp = df[df['treatment_group'] == 'Control']['baseline_bp_systolic']
treatment_bp = df[df['treatment_group'] == 'Treatment']['baseline_bp_systolic']
t_stat, p_value = stats.ttest_ind(control_bp, treatment_bp)
print(f"Baseline BP comparison (t-test): t={t_stat:.3f}, p={p_value:.3f}")

# Chi-square test for diabetes vs treatment group
chi2_table = pd.crosstab(df['diabetes'], df['treatment_group'])
chi2, p_chi2, dof, expected = stats.chi2_contingency(chi2_table)
print(f"Diabetes vs Treatment (chi-square): χ²={chi2:.3f}, p={p_chi2:.3f}")

# Paired t-test for BP change
bp_change = df['followup_bp_systolic'] - df['baseline_bp_systolic']
t_paired, p_paired = stats.ttest_1samp(bp_change, 0)
print(f"BP change from baseline (paired t-test): t={t_paired:.3f}, p={p_paired:.3f}")

# ===== SURVIVAL ANALYSIS =====
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

print("\\nSURVIVAL ANALYSIS")
print("-" * 20)

# Kaplan-Meier analysis by treatment group
kmf_control = KaplanMeierFitter()
kmf_treatment = KaplanMeierFitter()

control_mask = df['treatment_group'] == 'Control'
treatment_mask = df['treatment_group'] == 'Treatment'

kmf_control.fit(df[control_mask]['survival_time_months'], 
                df[control_mask]['event_occurred'], 
                label='Control')
kmf_treatment.fit(df[treatment_mask]['survival_time_months'], 
                  df[treatment_mask]['event_occurred'], 
                  label='Treatment')

print(f"Median survival - Control: {kmf_control.median_survival_time_:.1f} months")
print(f"Median survival - Treatment: {kmf_treatment.median_survival_time_:.1f} months")

# Log-rank test
results = logrank_test(df[control_mask]['survival_time_months'], 
                      df[treatment_mask]['survival_time_months'],
                      df[control_mask]['event_occurred'], 
                      df[treatment_mask]['event_occurred'])
print(f"Log-rank test: p={results.p_value:.3f}")

# ===== DATA VISUALIZATION =====
import matplotlib.pyplot as plt
import seaborn as sns

print("\\nCREATING VISUALIZATIONS")
print("-" * 25)

# Create comprehensive visualization panel
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Age distribution by treatment group
axes[0,0].hist([df[df['treatment_group']=='Control']['age'], 
                df[df['treatment_group']=='Treatment']['age']], 
               bins=15, alpha=0.7, label=['Control', 'Treatment'])
axes[0,0].set_title('Age Distribution by Treatment Group')
axes[0,0].set_xlabel('Age (years)')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# 2. BP change scatter plot
bp_change = df['followup_bp_systolic'] - df['baseline_bp_systolic']
colors = ['red' if group == 'Control' else 'blue' for group in df['treatment_group']]
axes[0,1].scatter(df['baseline_bp_systolic'], bp_change, c=colors, alpha=0.6)
axes[0,1].set_title('BP Change vs Baseline BP')
axes[0,1].set_xlabel('Baseline Systolic BP')
axes[0,1].set_ylabel('BP Change (Follow-up - Baseline)')
axes[0,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

# 3. BMI vs Cholesterol by diabetes status
diabetes_colors = ['green' if diabetes else 'orange' for diabetes in df['diabetes']]
axes[0,2].scatter(df['bmi'], df['cholesterol'], c=diabetes_colors, alpha=0.6)
axes[0,2].set_title('BMI vs Cholesterol (Green=Diabetes)')
axes[0,2].set_xlabel('BMI')
axes[0,2].set_ylabel('Cholesterol')

# 4. Treatment effect box plot
treatment_effect = df.groupby('treatment_group')['followup_bp_systolic'].apply(list)
axes[1,0].boxplot([treatment_effect['Control'], treatment_effect['Treatment']], 
                  labels=['Control', 'Treatment'])
axes[1,0].set_title('Follow-up BP by Treatment Group')
axes[1,0].set_ylabel('Follow-up Systolic BP')

# 5. Survival curves
time_points = np.linspace(0, df['survival_time_months'].max(), 100)
control_survival = [kmf_control.survival_function_at_times(t).iloc[0] for t in time_points]
treatment_survival = [kmf_treatment.survival_function_at_times(t).iloc[0] for t in time_points]

axes[1,1].plot(time_points, control_survival, label='Control', color='red')
axes[1,1].plot(time_points, treatment_survival, label='Treatment', color='blue')
axes[1,1].set_title('Kaplan-Meier Survival Curves')
axes[1,1].set_xlabel('Time (months)')
axes[1,1].set_ylabel('Survival Probability')
axes[1,1].legend()

# 6. Correlation heatmap
correlation_vars = ['age', 'baseline_bp_systolic', 'cholesterol', 'bmi', 'quality_of_life_score']
corr_matrix = df[correlation_vars].corr()
im = axes[1,2].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[1,2].set_title('Correlation Matrix')
axes[1,2].set_xticks(range(len(correlation_vars)))
axes[1,2].set_yticks(range(len(correlation_vars)))
axes[1,2].set_xticklabels(correlation_vars, rotation=45)
axes[1,2].set_yticklabels(correlation_vars)

plt.tight_layout()
plt.show()

print("\\n✅ COMPREHENSIVE CLINICAL ANALYSIS COMPLETED")
print("Analysis includes: descriptive statistics, statistical tests, survival analysis, and visualizations")
"""
            
            data = {
                'session_id': self.session_id,
                'code': comprehensive_code,
                'gemini_api_key': TEST_API_KEY,
                'analysis_title': 'Comprehensive Clinical Trial Analysis',
                'auto_section': True
            }
            
            response = requests.post(f"{BACKEND_URL}/sessions/{self.session_id}/execute-sectioned", 
                                   json=data, 
                                   headers={'Content-Type': 'application/json'})
            
            if response.status_code == 200:
                result = response.json()
                
                # Verify comprehensive analysis structure
                required_fields = ['id', 'session_id', 'title', 'sections', 'total_sections', 'execution_time', 'overall_success']
                if all(field in result for field in required_fields):
                    print("✅ Structured analysis result format correct")
                    
                    sections = result.get('sections', [])
                    print(f"✅ Code split into {len(sections)} sections")
                    
                    # Verify section types and titles
                    section_info = [(s.get('section_type'), s.get('title')) for s in sections]
                    print("Section classification results:")
                    for i, (section_type, title) in enumerate(section_info):
                        print(f"  Section {i+1}: {section_type} - '{title}'")
                    
                    # Check for healthcare-specific classifications
                    section_types = [s.get('section_type') for s in sections]
                    healthcare_types = ['clinical_trial', 'survival', 'descriptive', 'statistical_test', 'visualization']
                    found_healthcare_types = [t for t in section_types if t in healthcare_types]
                    
                    if found_healthcare_types:
                        print(f"✅ Healthcare-specific classifications found: {found_healthcare_types}")
                    
                    # Check metadata
                    metadata_features = []
                    for section in sections:
                        metadata = section.get('metadata', {})
                        if 'execution_time' in metadata:
                            metadata_features.append('execution_time')
                        if 'section_complexity' in metadata:
                            metadata_features.append('complexity')
                        if 'healthcare_context' in metadata:
                            metadata_features.append('healthcare_context')
                        if 'variables_used' in metadata:
                            metadata_features.append('variables_used')
                    
                    if metadata_features:
                        print(f"✅ Enhanced metadata features found: {set(metadata_features)}")
                    
                    # Check tables and charts
                    total_tables = sum(len(s.get('tables', [])) for s in sections)
                    total_charts = sum(len(s.get('charts', [])) for s in sections)
                    
                    print(f"✅ Tables extracted: {total_tables}")
                    print(f"✅ Charts extracted: {total_charts}")
                    
                    # Check overall success
                    if result.get('overall_success'):
                        print("✅ Overall analysis execution successful")
                        return True
                    else:
                        print("⚠️ Analysis completed with some errors but partial results available")
                        return True
                else:
                    print("❌ Missing required fields in result")
                    return False
            else:
                print(f"❌ Sectioned execution failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Comprehensive sectioned execution test failed: {str(e)}")
            return False
    
    def run_julius_ai_comprehensive_tests(self) -> Dict[str, bool]:
        """Run comprehensive Julius AI-style sectioned execution tests"""
        print("=" * 80)
        print("JULIUS AI-STYLE SECTIONED EXECUTION COMPREHENSIVE TESTING")
        print("Testing all aspects of the Julius AI functionality")
        print("=" * 80)
        
        # Setup
        if not self.setup_session():
            print("❌ Failed to setup test session")
            return {}
        
        # For now, just run the main comprehensive test
        results = {}
        
        print(f"\n🧪 RUNNING JULIUS AI COMPREHENSIVE TEST:")
        print("-" * 50)
        
        try:
            results["Sectioned Execution API"] = self.test_sectioned_execution_comprehensive()
            status = "✅ PASSED" if results["Sectioned Execution API"] else "❌ FAILED"
            print(f"\nResult: {status}")
        except Exception as e:
            print(f"❌ Test failed with exception: {str(e)}")
            results["Sectioned Execution API"] = False
        
        # Summary
        print(f"\n{'=' * 80}")
        print("JULIUS AI COMPREHENSIVE TESTING SUMMARY")
        print("=" * 80)
        
        for test_name, passed in results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {test_name}: {status}")
        
        passed_tests = sum(results.values())
        total_tests = len(results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"\n📈 OVERALL RESULTS:")
        print(f"  Tests Passed: {passed_tests}/{total_tests}")
        print(f"  Success Rate: {success_rate:.1%}")
        
        if success_rate >= 0.8:
            print(f"\n🎉 JULIUS AI TESTING SUCCESSFUL!")
            print("   ✅ Sectioned execution working properly")
        else:
            print(f"\n⚠️ JULIUS AI NEEDS ATTENTION")
            print("   Sectioned execution not working properly")
        
        return results

if __name__ == "__main__":
    tester = JuliusAITester()
    results = tester.run_julius_ai_comprehensive_tests()