# API Reference

## Base URL
```
http://localhost:8001/api
```

## Authentication
- **Gemini API Key**: Required for LLM endpoints, passed in request body
- **No Authentication**: Required for other endpoints (local development)

## Error Handling

### Standard Error Response
```json
{
  "detail": "Error message",
  "status_code": 400,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Common HTTP Status Codes
- `200` - Success
- `400` - Bad Request (validation error)
- `401` - Unauthorized (invalid API key)
- `404` - Not Found (resource doesn't exist)
- `422` - Validation Error (invalid data format)
- `429` - Rate Limit Exceeded
- `500` - Internal Server Error

## Endpoints

### Health Check

#### GET /health
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

### Sessions

#### GET /sessions
List all analysis sessions.

**Response:**
```json
[
  {
    "id": "session-uuid",
    "filename": "medical_data.csv",
    "upload_date": "2024-01-01T10:00:00Z",
    "row_count": 1000,
    "column_count": 15,
    "data_preview": {
      "columns": ["patient_id", "age", "gender", "diagnosis"],
      "sample_data": [
        {"patient_id": "P001", "age": 45, "gender": "F", "diagnosis": "Hypertension"}
      ]
    }
  }
]
```

#### POST /sessions
Create a new analysis session with CSV upload.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: `file` (CSV file)

**Response:**
```json
{
  "id": "session-uuid",
  "filename": "medical_data.csv",
  "upload_date": "2024-01-01T10:00:00Z",
  "row_count": 1000,
  "column_count": 15,
  "data_preview": {
    "columns": ["patient_id", "age", "gender", "diagnosis"],
    "dtypes": {
      "patient_id": "object",
      "age": "int64",
      "gender": "object",
      "diagnosis": "object"
    },
    "null_counts": {
      "patient_id": 0,
      "age": 5,
      "gender": 2,
      "diagnosis": 0
    },
    "statistics": {
      "age": {
        "mean": 52.3,
        "std": 15.2,
        "min": 18,
        "max": 89
      }
    }
  }
}
```

#### GET /sessions/{session_id}
Get details of a specific session.

**Parameters:**
- `session_id` (path): UUID of the session

**Response:**
```json
{
  "id": "session-uuid",
  "filename": "medical_data.csv",
  "upload_date": "2024-01-01T10:00:00Z",
  "row_count": 1000,
  "column_count": 15,
  "data_preview": {
    "columns": ["patient_id", "age", "gender", "diagnosis"],
    "sample_data": [...]
  }
}
```

#### GET /sessions/{session_id}/messages
Get chat messages for a session.

**Parameters:**
- `session_id` (path): UUID of the session

**Response:**
```json
[
  {
    "id": "message-uuid",
    "session_id": "session-uuid",
    "role": "user",
    "content": "What statistical tests would you recommend?",
    "timestamp": "2024-01-01T10:05:00Z"
  },
  {
    "id": "message-uuid",
    "session_id": "session-uuid",
    "role": "assistant",
    "content": "Based on your data, I recommend...",
    "timestamp": "2024-01-01T10:05:30Z"
  }
]
```

---

### Analysis

#### POST /sessions/{session_id}/chat
Chat with the LLM for analysis guidance.

**Parameters:**
- `session_id` (path): UUID of the session

**Request Body:**
```json
{
  "message": "What statistical tests would you recommend for this data?",
  "gemini_api_key": "your_gemini_api_key"
}
```

**Response:**
```json
{
  "response": "Based on your dataset with 1000 patients and 15 variables, I recommend the following statistical approaches:\n\n1. **Descriptive Statistics**: Start with df.describe() to understand your data distribution...",
  "message_id": "message-uuid"
}
```

#### POST /sessions/{session_id}/execute
Execute Python code in the analysis sandbox.

**Parameters:**
- `session_id` (path): UUID of the session

**Request Body:**
```json
{
  "code": "import pandas as pd\nimport numpy as np\n\n# Basic descriptive statistics\nprint(df.describe())\n\n# Create a simple plot\nimport matplotlib.pyplot as plt\nplt.figure(figsize=(10, 6))\ndf['age'].hist(bins=30)\nplt.title('Age Distribution')\nplt.xlabel('Age')\nplt.ylabel('Frequency')\nplt.show()"
}
```

**Response:**
```json
{
  "output": "                age\ncount  1000.000000\nmean     52.300000\nstd      15.200000\nmin      18.000000\n25%      41.000000\n50%      52.000000\n75%      63.000000\nmax      89.000000",
  "plots": [
    {
      "type": "image/png",
      "data": "iVBORw0KGgoAAAANSUhEUgAAAr4AAAH5CAYAAABOqx3K...",
      "title": "Age Distribution"
    }
  ],
  "success": true,
  "error": null,
  "execution_time": 0.245
}
```

#### POST /sessions/{session_id}/execute-sectioned
Execute code with Julius AI-style automatic sectioning.

**Parameters:**
- `session_id` (path): UUID of the session

**Request Body:**
```json
{
  "code": "# Data Overview\nprint('Dataset shape:', df.shape)\nprint('Column names:', df.columns.tolist())\n\n# Descriptive Statistics\nprint(df.describe())\n\n# Statistical Test\nfrom scipy import stats\nresult = stats.ttest_ind(df[df['gender']=='M']['age'], df[df['gender']=='F']['age'])\nprint(f'T-test result: t={result.statistic:.3f}, p={result.pvalue:.3f}')\n\n# Visualization\nimport matplotlib.pyplot as plt\nplt.figure(figsize=(10, 6))\ndf.boxplot(column='age', by='gender')\nplt.title('Age Distribution by Gender')\nplt.show()"
}
```

**Response:**
```json
{
  "id": "analysis-uuid",
  "session_id": "session-uuid",
  "title": "Healthcare Data Analysis",
  "sections": [
    {
      "id": "section-1",
      "title": "Clinical Data Overview",
      "section_type": "summary",
      "code": "print('Dataset shape:', df.shape)\nprint('Column names:', df.columns.tolist())",
      "output": "Dataset shape: (1000, 15)\nColumn names: ['patient_id', 'age', 'gender', 'diagnosis', ...]",
      "success": true,
      "error": null,
      "metadata": {
        "execution_time": 0.012,
        "complexity": "low",
        "healthcare_context": "clinical_data"
      },
      "tables": [],
      "charts": [],
      "order": 1
    },
    {
      "id": "section-2",
      "title": "Descriptive Statistics",
      "section_type": "descriptive",
      "code": "print(df.describe())",
      "output": "Age statistics and distribution summary...",
      "success": true,
      "error": null,
      "metadata": {
        "execution_time": 0.023,
        "complexity": "low",
        "healthcare_context": "general_healthcare"
      },
      "tables": [
        {
          "title": "Descriptive Statistics",
          "type": "statistical_results",
          "data": "Table data..."
        }
      ],
      "charts": [],
      "order": 2
    },
    {
      "id": "section-3",
      "title": "Statistical Analysis",
      "section_type": "statistical_test",
      "code": "from scipy import stats\nresult = stats.ttest_ind(df[df['gender']=='M']['age'], df[df['gender']=='F']['age'])\nprint(f'T-test result: t={result.statistic:.3f}, p={result.pvalue:.3f}')",
      "output": "T-test result: t=1.245, p=0.213",
      "success": true,
      "error": null,
      "metadata": {
        "execution_time": 0.089,
        "complexity": "medium",
        "healthcare_context": "clinical_research"
      },
      "tables": [],
      "charts": [],
      "order": 3
    },
    {
      "id": "section-4",
      "title": "Data Visualization",
      "section_type": "visualization",
      "code": "import matplotlib.pyplot as plt\nplt.figure(figsize=(10, 6))\ndf.boxplot(column='age', by='gender')\nplt.title('Age Distribution by Gender')\nplt.show()",
      "output": "",
      "success": true,
      "error": null,
      "metadata": {
        "execution_time": 0.456,
        "complexity": "medium",
        "healthcare_context": "clinical_research"
      },
      "tables": [],
      "charts": [
        {
          "type": "box_plot",
          "title": "Age Distribution by Gender",
          "data": "base64_encoded_image_data"
        }
      ],
      "order": 4
    }
  ],
  "total_sections": 4,
  "execution_time": 0.580,
  "timestamp": "2024-01-01T10:15:00Z",
  "overall_success": true
}
```

#### POST /sessions/{session_id}/suggest-analysis
Get AI-powered analysis suggestions for the dataset.

**Parameters:**
- `session_id` (path): UUID of the session

**Request Body:**
```json
{
  "gemini_api_key": "your_gemini_api_key"
}
```

**Response:**
```json
{
  "suggestions": "Based on your medical dataset with 1000 patients and variables including age, gender, and diagnosis, here are my recommendations:\n\n## Recommended Analysis Approaches\n\n### 1. Descriptive Statistics\n- Age distribution analysis\n- Gender distribution\n- Diagnosis frequency analysis\n\n### 2. Statistical Tests\n- Chi-square test for categorical associations\n- T-test for age differences by gender\n- ANOVA for diagnosis group comparisons\n\n### 3. Visualization\n- Age histogram by gender\n- Diagnosis frequency bar chart\n- Correlation heatmap for numeric variables\n\n### 4. Advanced Analysis\n- Survival analysis if time-to-event data available\n- Logistic regression for outcome prediction\n- Clustering analysis for patient stratification"
}
```

---

### Structured Analysis

#### GET /sessions/{session_id}/structured-analyses
List all structured analyses for a session.

**Parameters:**
- `session_id` (path): UUID of the session

**Response:**
```json
[
  {
    "id": "analysis-uuid",
    "session_id": "session-uuid",
    "title": "Healthcare Data Analysis",
    "total_sections": 4,
    "execution_time": 0.580,
    "timestamp": "2024-01-01T10:15:00Z",
    "overall_success": true
  }
]
```

#### GET /sessions/{session_id}/structured-analyses/{analysis_id}
Get complete details of a specific structured analysis.

**Parameters:**
- `session_id` (path): UUID of the session
- `analysis_id` (path): UUID of the analysis

**Response:**
```json
{
  "id": "analysis-uuid",
  "session_id": "session-uuid",
  "title": "Healthcare Data Analysis",
  "sections": [
    {
      "id": "section-1",
      "title": "Clinical Data Overview",
      "section_type": "summary",
      "code": "print('Dataset shape:', df.shape)",
      "output": "Dataset shape: (1000, 15)",
      "success": true,
      "error": null,
      "metadata": {
        "execution_time": 0.012,
        "complexity": "low",
        "healthcare_context": "clinical_data",
        "variables_used": ["df"],
        "data_modifications": []
      },
      "tables": [],
      "charts": [],
      "order": 1
    }
  ],
  "total_sections": 4,
  "execution_time": 0.580,
  "timestamp": "2024-01-01T10:15:00Z",
  "overall_success": true
}
```

---

### Analysis History

#### GET /sessions/{session_id}/analysis-history
Get analysis history for a session.

**Parameters:**
- `session_id` (path): UUID of the session

**Response:**
```json
[
  {
    "id": "history-uuid",
    "session_id": "session-uuid",
    "analysis_type": "t-test",
    "description": "Independent t-test comparing age between genders",
    "results": {
      "test_statistic": 1.245,
      "p_value": 0.213,
      "effect_size": 0.089,
      "confidence_interval": [-0.5, 2.1],
      "interpretation": "No significant difference in age between genders (p=0.213)"
    },
    "timestamp": "2024-01-01T10:10:00Z"
  }
]
```

---

## Code Examples

### Python Client
```python
import requests
import json

# Base configuration
BASE_URL = "http://localhost:8001/api"
GEMINI_API_KEY = "your_gemini_api_key"

# Upload CSV and create session
def upload_csv(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{BASE_URL}/sessions", files=files)
    return response.json()

# Chat with LLM
def chat_with_llm(session_id, message):
    data = {
        "message": message,
        "gemini_api_key": GEMINI_API_KEY
    }
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/chat", json=data)
    return response.json()

# Execute Python code
def execute_code(session_id, code):
    data = {"code": code}
    response = requests.post(f"{BASE_URL}/sessions/{session_id}/execute", json=data)
    return response.json()

# Usage example
session = upload_csv("medical_data.csv")
chat_response = chat_with_llm(session['id'], "What tests should I run?")
code_result = execute_code(session['id'], "df.describe()")
```

### JavaScript Client
```javascript
const BASE_URL = "http://localhost:8001/api";
const GEMINI_API_KEY = "your_gemini_api_key";

// Upload CSV and create session
async function uploadCSV(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${BASE_URL}/sessions`, {
    method: 'POST',
    body: formData,
  });
  
  return await response.json();
}

// Chat with LLM
async function chatWithLLM(sessionId, message) {
  const response = await fetch(`${BASE_URL}/sessions/${sessionId}/chat`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message: message,
      gemini_api_key: GEMINI_API_KEY,
    }),
  });
  
  return await response.json();
}

// Execute Python code
async function executeCode(sessionId, code) {
  const response = await fetch(`${BASE_URL}/sessions/${sessionId}/execute`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      code: code,
    }),
  });
  
  return await response.json();
}
```

### cURL Examples
```bash
# Upload CSV
curl -X POST "http://localhost:8001/api/sessions" \
  -F "file=@medical_data.csv"

# Chat with LLM
curl -X POST "http://localhost:8001/api/sessions/{session_id}/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What statistical tests would you recommend?",
    "gemini_api_key": "your_gemini_api_key"
  }'

# Execute Python code
curl -X POST "http://localhost:8001/api/sessions/{session_id}/execute" \
  -H "Content-Type: application/json" \
  -d '{
    "code": "df.describe()"
  }'

# Get structured analyses
curl -X GET "http://localhost:8001/api/sessions/{session_id}/structured-analyses"
```

## Rate Limits

- **Gemini API**: Subject to Google's rate limits
- **File Upload**: Max 100MB per file
- **Code Execution**: 30 second timeout per execution
- **Session Storage**: 30 days retention

## Best Practices

1. **Always validate API responses** before using data
2. **Handle rate limits gracefully** with exponential backoff
3. **Use structured analysis** for complex workflows
4. **Store analysis IDs** for result retrieval
5. **Monitor execution times** for optimization opportunities