#!/usr/bin/env python3
"""
Simple test to isolate the JSON serialization issue in sectioned execution
"""

import requests
import json
import pandas as pd

# Configuration
BACKEND_URL = "https://ebc2c5fe-7f5a-4ab1-82f3-4604f56fc650.preview.emergentagent.com/api"
TEST_API_KEY = "test_key_123"

def test_simple_sectioned_execution():
    """Test simple sectioned execution to isolate JSON serialization issue"""
    
    # First create a session
    print("Creating test session...")
    simple_csv = "age,gender\n25,Male\n30,Female\n35,Male"
    files = {'file': ('test.csv', simple_csv, 'text/csv')}
    
    response = requests.post(f"{BACKEND_URL}/sessions", files=files, timeout=30)
    if response.status_code != 200:
        print(f"❌ Session creation failed: {response.status_code}")
        return False
    
    session_id = response.json().get('id')
    print(f"✅ Session created: {session_id}")
    
    # Test very simple code
    simple_code = """
print("Hello World")
print("Dataset shape:", df.shape)
"""
    
    data = {
        'session_id': session_id,
        'code': simple_code,
        'gemini_api_key': TEST_API_KEY,
        'analysis_title': 'Simple Test',
        'auto_section': True
    }
    
    print("Testing simple sectioned execution...")
    response = requests.post(f"{BACKEND_URL}/sessions/{session_id}/execute-sectioned", 
                           json=data, 
                           headers={'Content-Type': 'application/json'})
    
    print(f"Response status: {response.status_code}")
    if response.status_code != 200:
        print(f"Response text: {response.text}")
        return False
    else:
        result = response.json()
        print("✅ Simple sectioned execution successful")
        print(f"Sections: {len(result.get('sections', []))}")
        return True

if __name__ == "__main__":
    test_simple_sectioned_execution()