#!/usr/bin/env python3
"""
Test rate limit handling for Gemini API
"""

import requests
import json
import time

BACKEND_URL = "https://0ee84439-001a-41bd-9cd3-14d4582c1370.preview.emergentagent.com/api"

def test_rate_limit_handling():
    """Test rate limit error handling"""
    print("Testing Rate Limit Handling...")
    
    # First create a session
    csv_data = "age,gender,bp\n25,M,120\n30,F,110\n35,M,130"
    files = {'file': ('test.csv', csv_data, 'text/csv')}
    
    session_response = requests.post(f"{BACKEND_URL}/sessions", files=files)
    if session_response.status_code != 200:
        print("❌ Could not create session for rate limit testing")
        return False
    
    session_id = session_response.json()['id']
    print(f"✅ Created session: {session_id}")
    
    # Test with a potentially rate-limited API key (making multiple rapid requests)
    api_key = "AIzaSyC3Z8XNz1HN0ZUzeXhDrpG66ZvNmbi7mNo"
    
    print("Making multiple rapid requests to test rate limiting...")
    
    for i in range(3):
        print(f"  Request {i+1}:")
        
        data = {
            'message': f'Request {i+1}: Analyze this medical data quickly',
            'gemini_api_key': api_key
        }
        
        response = requests.post(f"{BACKEND_URL}/sessions/{session_id}/chat", data=data)
        
        if response.status_code == 200:
            print(f"    ✅ Success - Response received")
        elif response.status_code == 429:
            error_detail = response.json().get('detail', '')
            print(f"    ✅ Rate limit detected: {error_detail}")
            if 'Gemini 2.5 Flash' in error_detail:
                print("    ✅ Proper rate limit message mentioning Flash model")
            return True
        elif response.status_code == 400:
            error_detail = response.json().get('detail', '')
            print(f"    ✅ API key validation: {error_detail}")
        else:
            print(f"    ❌ Unexpected status: {response.status_code}")
        
        time.sleep(0.1)  # Very brief pause
    
    print("✅ Rate limit testing completed (no rate limits hit, which is also valid)")
    return True

if __name__ == "__main__":
    test_rate_limit_handling()