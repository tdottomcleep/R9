import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

console.log('Backend URL:', BACKEND_URL);
console.log('API URL:', API);

const App = () => {
  const [sessions, setSessions] = useState([]);
  const [currentSession, setCurrentSession] = useState(null);
  const [messages, setMessages] = useState([]);
  const [newMessage, setNewMessage] = useState('');
  const [geminiApiKey, setGeminiApiKey] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [executionResult, setExecutionResult] = useState(null);
  const [analysisHistory, setAnalysisHistory] = useState([]);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [activeResultTab, setActiveResultTab] = useState('execution'); // 'execution' or 'history'
  const [structuredAnalyses, setStructuredAnalyses] = useState([]);
  
  // Panel resizing state
  const [centerPanelWidth, setCenterPanelWidth] = useState(60); // percentage
  const [isDragging, setIsDragging] = useState(false);
  const [dragStartX, setDragStartX] = useState(0);
  const [dragStartWidth, setDragStartWidth] = useState(60);
  
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    if (currentSession) {
      fetchMessages(currentSession.id);
      fetchAnalysisHistory(currentSession.id);
      fetchStructuredAnalyses(currentSession.id);
    }
  }, [currentSession]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Handle panel dragging
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isDragging) return;
      
      const containerWidth = window.innerWidth - (leftPanelOpen ? 320 : 48) - (rightPanelOpen ? 48 : 48);
      const deltaX = e.clientX - dragStartX;
      const deltaPercentage = (deltaX / containerWidth) * 100;
      const newWidth = Math.min(Math.max(dragStartWidth + deltaPercentage, 30), 80); // Min 30%, Max 80%
      
      setCenterPanelWidth(newWidth);
    };

    const handleMouseUp = () => {
      setIsDragging(false);
      setDragStartX(0);
      setDragStartWidth(0);
      document.body.style.cursor = 'default';
      document.body.style.userSelect = 'auto';
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'col-resize';
      document.body.style.userSelect = 'none';
    }

    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
      document.body.style.cursor = 'default';
      document.body.style.userSelect = 'auto';
    };
  }, [isDragging, dragStartX, dragStartWidth, leftPanelOpen, rightPanelOpen]);

  const handleDragStart = (e) => {
    setIsDragging(true);
    setDragStartX(e.clientX);
    setDragStartWidth(centerPanelWidth);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchSessions = async () => {
    try {
      console.log('Fetching sessions from:', `${API}/sessions`);
      const response = await axios.get(`${API}/sessions`);
      console.log('Sessions response:', response.data);
      setSessions(response.data);
    } catch (error) {
      console.error('Error fetching sessions:', error);
      console.error('Fetch sessions error response:', error.response);
    }
  };

  const fetchMessages = async (sessionId) => {
    try {
      const response = await axios.get(`${API}/sessions/${sessionId}/messages`);
      setMessages(response.data);
    } catch (error) {
      console.error('Error fetching messages:', error);
    }
  };

  const fetchAnalysisHistory = async (sessionId) => {
    try {
      const response = await axios.get(`${API}/sessions/${sessionId}/analysis-history`);
      setAnalysisHistory(response.data);
    } catch (error) {
      console.error('Error fetching analysis history:', error);
    }
  };

  const handleFileUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith('.csv')) {
      alert('Please upload a CSV file');
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      console.log('Uploading file to:', `${API}/sessions`);
      console.log('FormData:', formData);
      console.log('File:', file);

      const response = await axios.post(`${API}/sessions`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      console.log('Upload response:', response);
      console.log('Response data:', response.data);

      await fetchSessions();
      setCurrentSession(response.data);
      setMessages([]);
      setExecutionResult(null);
    } catch (error) {
      console.error('Error uploading file:', error);
      console.error('Error response:', error.response);
      console.error('Error message:', error.message);
      console.error('Error status:', error.response?.status);
      console.error('Error data:', error.response?.data);
      
      let errorMessage = 'Error uploading file. Please try again.';
      if (error.response?.data?.detail) {
        errorMessage = `Upload failed: ${error.response.data.detail}`;
      } else if (error.response?.status) {
        errorMessage = `Upload failed with status ${error.response.status}`;
      }
      
      alert(errorMessage);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSendMessage = async () => {
    if (!newMessage.trim() || !currentSession) return;

    if (!geminiApiKey) {
      setShowApiKeyModal(true);
      return;
    }

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('message', newMessage);
      formData.append('gemini_api_key', geminiApiKey);

      // Add user message to UI immediately
      const userMessage = {
        id: Date.now().toString(),
        role: 'user',
        content: newMessage,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, userMessage]);
      setNewMessage('');

      const response = await axios.post(`${API}/sessions/${currentSession.id}/chat`, formData);
      
      // Add assistant response
      const assistantMessage = {
        id: Date.now().toString() + '_assistant',
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, assistantMessage]);
      
    } catch (error) {
      console.error('Error sending message:', error);
      alert('Error sending message. Please check your API key and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchStructuredAnalyses = async (sessionId) => {
    try {
      const response = await axios.get(`${API}/sessions/${sessionId}/structured-analyses`);
      setStructuredAnalyses(response.data);
    } catch (error) {
      console.error('Error fetching structured analyses:', error);
    }
  };

  const handleExecuteCode = async (code) => {
    if (!currentSession || !geminiApiKey) return;

    setIsLoading(true);
    try {
      // Use sectioned execution for Julius AI-style results
      const response = await axios.post(`${API}/sessions/${currentSession.id}/execute-sectioned`, {
        session_id: currentSession.id,
        code: code,
        gemini_api_key: geminiApiKey,
        analysis_title: "Statistical Analysis"
      });

      setExecutionResult(response.data);
      setRightPanelOpen(true);
      
      // Also fetch updated structured analyses
      fetchStructuredAnalyses(currentSession.id);
      
    } catch (error) {
      console.error('Error executing code:', error);
      setExecutionResult({
        overall_success: false,
        error: 'Error executing code: ' + error.message,
        sections: []
      });
    } finally {
      setIsLoading(false);
    }
  };

  const getAnalysisSuggestions = async () => {
    if (!currentSession || !geminiApiKey) return;

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append('gemini_api_key', geminiApiKey);

      const response = await axios.post(`${API}/sessions/${currentSession.id}/suggest-analysis`, formData);
      
      // Add suggestions as a system message
      const suggestionsMessage = {
        id: Date.now().toString() + '_suggestions',
        role: 'assistant',
        content: `🔬 **Statistical Analysis Suggestions**\n\n${response.data.suggestions}`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, suggestionsMessage]);
      
    } catch (error) {
      console.error('Error getting analysis suggestions:', error);
      alert('Error getting analysis suggestions. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const DataPreview = ({ session }) => {
    if (!session?.csv_preview) return null;

    const { columns, shape, head, dtypes, null_counts, describe } = session.csv_preview;

    // Analyze data types
    const numericCols = columns?.filter(col => 
      dtypes?.[col] && (dtypes[col].includes('int') || dtypes[col].includes('float'))
    ) || [];
    const categoricalCols = columns?.filter(col => 
      dtypes?.[col] && dtypes[col].includes('object')
    ) || [];

    return (
      <div className="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg mb-4 border border-blue-200">
        <h3 className="font-semibold text-gray-800 mb-3 flex items-center">
          <svg className="w-5 h-5 mr-2 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          Dataset Overview
        </h3>
        
        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-white p-3 rounded border">
            <p className="text-sm font-medium text-gray-700">Dataset Size</p>
            <p className="text-lg font-semibold text-blue-600">{shape?.[0]} × {shape?.[1]}</p>
            <p className="text-xs text-gray-500">rows × columns</p>
          </div>
          <div className="bg-white p-3 rounded border">
            <p className="text-sm font-medium text-gray-700">Data Quality</p>
            <p className="text-lg font-semibold text-green-600">
              {null_counts ? Math.round((1 - Object.values(null_counts).reduce((a, b) => a + b, 0) / (shape?.[0] * shape?.[1])) * 100) : 100}%
            </p>
            <p className="text-xs text-gray-500">completeness</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-4">
          <div className="bg-white p-3 rounded border">
            <p className="text-sm font-medium text-gray-700">Numeric Variables</p>
            <p className="text-lg font-semibold text-purple-600">{numericCols.length}</p>
            <p className="text-xs text-gray-500">{numericCols.slice(0, 3).join(', ')}{numericCols.length > 3 ? '...' : ''}</p>
          </div>
          <div className="bg-white p-3 rounded border">
            <p className="text-sm font-medium text-gray-700">Categorical Variables</p>
            <p className="text-lg font-semibold text-orange-600">{categoricalCols.length}</p>
            <p className="text-xs text-gray-500">{categoricalCols.slice(0, 3).join(', ')}{categoricalCols.length > 3 ? '...' : ''}</p>
          </div>
        </div>
        
        {head && head.length > 0 && (
          <div className="bg-white p-3 rounded border">
            <h4 className="font-medium text-gray-700 mb-2">Sample Data Preview</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="bg-gray-100">
                    {columns?.slice(0, 6).map(col => (
                      <th key={col} className="px-2 py-1 text-left font-medium">{col}</th>
                    ))}
                    {columns?.length > 6 && <th className="px-2 py-1 text-left">...</th>}
                  </tr>
                </thead>
                <tbody>
                  {head.slice(0, 3).map((row, idx) => (
                    <tr key={idx} className="border-b">
                      {columns?.slice(0, 6).map(col => (
                        <td key={col} className="px-2 py-1">{row[col]}</td>
                      ))}
                      {columns?.length > 6 && <td className="px-2 py-1">...</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  };

  const JuliusStyleCodeBlock = ({ code, onExecute, sectionTitle = "Code Analysis" }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const [showLineNumbers, setShowLineNumbers] = useState(true);
    
    const codeLines = code.split('\n');
    const displayLines = isExpanded ? codeLines : codeLines.slice(0, 8);
    const hasMoreLines = codeLines.length > 8;
    
    return (
      <div className="bg-white border border-gray-200 rounded-lg my-3 shadow-sm">
        {/* Header */}
        <div className="bg-gray-50 border-b border-gray-200 px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
            <h4 className="text-sm font-medium text-gray-700">{sectionTitle}</h4>
            <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded">
              {codeLines.length} lines
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowLineNumbers(!showLineNumbers)}
              className="text-xs text-gray-500 hover:text-gray-700 px-2 py-1 rounded hover:bg-gray-100"
            >
              {showLineNumbers ? 'Hide' : 'Show'} Lines
            </button>
            <button
              onClick={() => onExecute(code)}
              className="bg-blue-600 text-white px-3 py-1 rounded text-xs hover:bg-blue-700 transition-colors"
            >
              Execute
            </button>
          </div>
        </div>
        
        {/* Code Content */}
        <div className="relative">
          <div 
            className="overflow-auto text-sm bg-gray-50"
            style={{ maxHeight: '300px' }}
          >
            <div className="flex">
              {/* Line Numbers */}
              {showLineNumbers && (
                <div className="bg-gray-100 text-gray-500 px-3 py-2 border-r border-gray-200 select-none">
                  {displayLines.map((_, index) => (
                    <div key={index} className="text-right font-mono text-xs leading-5">
                      {index + 1}
                    </div>
                  ))}
                </div>
              )}
              
              {/* Code */}
              <div className="flex-1 px-4 py-2">
                <pre className="text-gray-800 font-mono text-xs leading-5 whitespace-pre-wrap">
                  {displayLines.join('\n')}
                </pre>
              </div>
            </div>
          </div>
          
          {/* Expand/Collapse Button */}
          {hasMoreLines && (
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-white to-transparent p-3 text-center">
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="text-blue-600 hover:text-blue-800 text-sm font-medium bg-white border border-gray-300 px-3 py-1 rounded shadow-sm hover:bg-gray-50 transition-colors"
              >
                {isExpanded ? 'Show Less' : `Show ${codeLines.length - 8} More Lines`}
              </button>
            </div>
          )}
        </div>
      </div>
    );
  };

  const CodeBlock = ({ code, onExecute }) => {
    return (
      <div className="bg-gray-900 text-green-400 p-3 rounded-lg my-2 relative">
        <pre className="text-sm overflow-x-auto">{code}</pre>
        <button
          onClick={() => onExecute(code)}
          className="absolute top-2 right-2 bg-blue-600 text-white px-2 py-1 rounded text-xs hover:bg-blue-700"
        >
          Execute
        </button>
      </div>
    );
  };

  const MessageRenderer = ({ message }) => {
    const isUser = message.role === 'user';
    const content = message.content;
    
    // Simple code block detection
    const codeBlocks = content.match(/```python\n([\s\S]*?)```/g);
    
    return (
      <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
        <div className={`max-w-3xl p-3 rounded-lg ${
          isUser 
            ? 'bg-blue-600 text-white' 
            : 'bg-gray-100 text-gray-800'
        }`}>
          {codeBlocks ? (
            <div>
              {content.split(/```python\n([\s\S]*?)```/).map((part, index) => {
                if (index % 2 === 0) {
                  return <p key={index} className="whitespace-pre-wrap">{part}</p>;
                } else {
                  return <JuliusStyleCodeBlock key={index} code={part} onExecute={handleExecuteCode} sectionTitle="Python Code" />;
                }
              })}
            </div>
          ) : (
            <p className="whitespace-pre-wrap">{content}</p>
          )}
        </div>
      </div>
    );
  };

  const AnalysisHistoryPanel = () => {
    return (
      <div className="h-full overflow-y-auto p-4 space-y-4">
        <h3 className="font-semibold text-gray-700 border-b pb-2">Analysis History</h3>
        
        {analysisHistory.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <svg className="mx-auto h-8 w-8 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            <p className="text-sm">No statistical analyses performed yet</p>
          </div>
        ) : (
          <div className="space-y-4">
            {analysisHistory.map((analysis, index) => (
              <div key={analysis.id} className="bg-white border rounded-lg p-4 shadow-sm">
                <div className="flex items-start justify-between mb-2">
                  <h4 className="font-medium text-gray-800">{analysis.analysis_type}</h4>
                  <span className="text-xs text-gray-500">
                    {new Date(analysis.timestamp).toLocaleString()}
                  </span>
                </div>
                
                <div className="text-sm text-gray-600 mb-2">
                  <strong>Variables:</strong> {analysis.variables.join(', ')}
                </div>
                
                {analysis.test_statistic && (
                  <div className="bg-gray-50 p-3 rounded text-sm">
                    <div className="grid grid-cols-2 gap-2">
                      <div>
                        <span className="font-medium">Test Statistic:</span> {analysis.test_statistic?.toFixed(4)}
                      </div>
                      <div>
                        <span className="font-medium">P-value:</span> 
                        <span className={analysis.p_value < 0.05 ? 'text-red-600 font-medium' : 'text-gray-700'}>
                          {analysis.p_value?.toFixed(4)}
                        </span>
                      </div>
                      {analysis.effect_size && (
                        <div>
                          <span className="font-medium">Effect Size:</span> {analysis.effect_size?.toFixed(4)}
                        </div>
                      )}
                      {analysis.confidence_interval && (
                        <div>
                          <span className="font-medium">95% CI:</span> [{analysis.confidence_interval[0]?.toFixed(4)}, {analysis.confidence_interval[1]?.toFixed(4)}]
                        </div>
                      )}
                    </div>
                  </div>
                )}
                
                <div className="mt-2 text-sm">
                  <span className="font-medium">Interpretation:</span> {analysis.interpretation}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  };

  const ExecutionPanel = () => {
    if (!executionResult) return null;

    return (
      <div className="h-full overflow-y-auto p-4 space-y-4">
        <div className="flex items-center justify-between border-b pb-3">
          <h3 className="font-semibold text-gray-700">Execution Results</h3>
          <div className="flex items-center space-x-2">
            {executionResult.overall_success ? (
              <span className="text-green-600 text-sm bg-green-50 px-2 py-1 rounded">
                ✅ Success
              </span>
            ) : (
              <span className="text-red-600 text-sm bg-red-50 px-2 py-1 rounded">
                ❌ Failed
              </span>
            )}
            {executionResult.total_sections && (
              <span className="text-gray-500 text-sm bg-gray-100 px-2 py-1 rounded">
                {executionResult.total_sections} sections
              </span>
            )}
          </div>
        </div>
        
        {/* Julius AI Style Sectioned Results */}
        {executionResult.sections && executionResult.sections.length > 0 ? (
          <div className="space-y-4">
            {executionResult.sections.map((section, index) => (
              <div key={section.id || index} className="bg-white border border-gray-200 rounded-lg shadow-sm">
                {/* Section Header */}
                <div className="bg-gray-50 border-b border-gray-200 px-4 py-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className={`w-3 h-3 rounded-full ${
                        section.section_type === 'descriptive' ? 'bg-blue-500' :
                        section.section_type === 'statistical_test' ? 'bg-green-500' :
                        section.section_type === 'visualization' ? 'bg-purple-500' :
                        section.section_type === 'model' ? 'bg-orange-500' :
                        'bg-gray-500'
                      }`}></div>
                      <h4 className="text-sm font-medium text-gray-700">{section.title}</h4>
                      <span className="text-xs text-gray-500 bg-gray-100 px-2 py-1 rounded capitalize">
                        {section.section_type?.replace('_', ' ')}
                      </span>
                    </div>
                    <div className="text-xs text-gray-500">
                      {section.success ? '✅' : '❌'}
                    </div>
                  </div>
                </div>
                
                {/* Section Content */}
                <div className="p-4">
                  {/* Code */}
                  <div className="mb-4">
                    <div className="bg-gray-50 border border-gray-200 rounded-lg overflow-hidden">
                      <div className="bg-gray-100 px-3 py-2 border-b border-gray-200">
                        <span className="text-xs font-medium text-gray-600">Code</span>
                      </div>
                      <div className="p-3 max-h-60 overflow-auto">
                        <pre className="text-sm text-gray-800 font-mono whitespace-pre-wrap">
                          {section.code}
                        </pre>
                      </div>
                    </div>
                  </div>
                  
                  {/* Output */}
                  {section.output && (
                    <div className="mb-4">
                      <div className="bg-gray-50 border border-gray-200 rounded-lg overflow-hidden">
                        <div className="bg-gray-100 px-3 py-2 border-b border-gray-200">
                          <span className="text-xs font-medium text-gray-600">Output</span>
                        </div>
                        <div className="p-3 max-h-60 overflow-auto">
                          <pre className="text-sm text-gray-800 whitespace-pre-wrap">
                            {section.output}
                          </pre>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* Error */}
                  {section.error && (
                    <div className="mb-4">
                      <div className="bg-red-50 border border-red-200 rounded-lg overflow-hidden">
                        <div className="bg-red-100 px-3 py-2 border-b border-red-200">
                          <span className="text-xs font-medium text-red-600">Error</span>
                        </div>
                        <div className="p-3 max-h-60 overflow-auto">
                          <pre className="text-sm text-red-700 whitespace-pre-wrap">
                            {section.error}
                          </pre>
                        </div>
                      </div>
                    </div>
                  )}
                  
                  {/* Charts */}
                  {section.charts && section.charts.length > 0 && (
                    <div className="mb-4">
                      <h5 className="text-sm font-medium text-gray-700 mb-2">Visualizations</h5>
                      <div className="grid grid-cols-1 gap-4">
                        {section.charts.map((chart, chartIndex) => (
                          <div key={chartIndex} className="bg-white border border-gray-200 rounded-lg p-4">
                            {chart.type === 'matplotlib' && chart.data ? (
                              <img 
                                src={`data:image/png;base64,${chart.data}`} 
                                alt={chart.title || `Chart ${chartIndex + 1}`}
                                className="max-w-full rounded shadow-sm"
                              />
                            ) : chart.type === 'plotly' && chart.html ? (
                              <div 
                                dangerouslySetInnerHTML={{ __html: chart.html }}
                                className="plotly-container"
                              />
                            ) : (
                              <p className="text-gray-500 text-sm">Chart data not available</p>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Tables */}
                  {section.tables && section.tables.length > 0 && (
                    <div className="mb-4">
                      <h5 className="text-sm font-medium text-gray-700 mb-2">Tables</h5>
                      <div className="space-y-4">
                        {section.tables.map((table, tableIndex) => (
                          <div key={tableIndex} className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                            <div className="bg-gray-50 px-3 py-2 border-b border-gray-200">
                              <span className="text-xs font-medium text-gray-600">
                                {table.title || `Table ${tableIndex + 1}`}
                              </span>
                            </div>
                            <div className="p-3 overflow-auto">
                              <pre className="text-sm text-gray-800 whitespace-pre-wrap">
                                {table.content}
                              </pre>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                  
                  {/* Metadata */}
                  {section.metadata && Object.keys(section.metadata).length > 0 && (
                    <div className="text-xs text-gray-500 border-t pt-3">
                      <div className="flex flex-wrap gap-4">
                        {section.metadata.execution_time && (
                          <span>⏱️ {section.metadata.execution_time}s</span>
                        )}
                        {section.metadata.complexity && (
                          <span>📊 {section.metadata.complexity}</span>
                        )}
                        {section.metadata.context && (
                          <span>🏥 {section.metadata.context}</span>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <svg className="mx-auto h-8 w-8 text-gray-400 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="text-sm">No execution results yet</p>
          </div>
        )}
      </div>
    );
  };

  const ApiKeyModal = () => {
    const [tempApiKey, setTempApiKey] = useState('');
    
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white p-6 rounded-lg max-w-md w-full mx-4">
          <h2 className="text-xl font-semibold mb-4">Gemini API Key Required</h2>
          <p className="text-gray-600 mb-4">
            Please enter your Gemini API key to continue. You can get it from the Google AI Studio.
          </p>
          <input
            type="password"
            value={tempApiKey}
            onChange={(e) => setTempApiKey(e.target.value)}
            placeholder="Enter your Gemini API key"
            className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <div className="flex justify-end space-x-2 mt-4">
            <button
              onClick={() => setShowApiKeyModal(false)}
              className="px-4 py-2 text-gray-600 hover:text-gray-800"
            >
              Cancel
            </button>
            <button
              onClick={() => {
                setGeminiApiKey(tempApiKey);
                setShowApiKeyModal(false);
              }}
              className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
            >
              Save
            </button>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="h-screen flex bg-gray-100">
      {/* Left Panel */}
      <div className={`${leftPanelOpen ? 'w-80' : 'w-12'} transition-all duration-300 bg-white shadow-lg flex flex-col`}>
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <h1 className={`font-bold text-xl text-gray-800 ${leftPanelOpen ? 'block' : 'hidden'}`}>
            AI Data Scientist
          </h1>
          <button
            onClick={() => setLeftPanelOpen(!leftPanelOpen)}
            className="p-2 hover:bg-gray-100 rounded"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
        
        {leftPanelOpen && (
          <>
            <div className="p-4 border-b border-gray-200">
              <input
                type="file"
                accept=".csv"
                onChange={handleFileUpload}
                ref={fileInputRef}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
              >
                {isLoading ? 'Uploading...' : 'Upload CSV File'}
              </button>
            </div>
            
            <div className="flex-1 overflow-y-auto">
              <div className="p-4">
                <h3 className="font-semibold text-gray-700 mb-2">Chat Sessions</h3>
                {sessions.length === 0 ? (
                  <p className="text-gray-500 text-sm">No sessions yet. Upload a CSV file to start.</p>
                ) : (
                  <div className="space-y-2">
                    {sessions.map(session => (
                      <div
                        key={session.id}
                        onClick={() => setCurrentSession(session)}
                        className={`p-3 rounded cursor-pointer transition-colors ${
                          currentSession?.id === session.id
                            ? 'bg-blue-100 border-l-4 border-blue-500'
                            : 'bg-gray-50 hover:bg-gray-100'
                        }`}
                      >
                        <div className="font-medium text-sm truncate">{session.title}</div>
                        <div className="text-xs text-gray-500">
                          {new Date(session.created_at).toLocaleDateString()}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </>
        )}
      </div>

      {/* Center Panel */}
      <div 
        className="flex flex-col bg-white"
        style={{ 
          width: rightPanelOpen ? `${centerPanelWidth}%` : 'calc(100% - 48px)'
        }}
      >
        {currentSession ? (
          <>
            <div className="bg-white border-b border-gray-200 p-4">
              <h2 className="font-semibold text-gray-800">{currentSession.title}</h2>
              <div className="text-sm text-gray-600">
                Chat with your data • {geminiApiKey ? '🟢 API Key Set' : '🔴 API Key Required'}
              </div>
            </div>
            
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              <DataPreview session={currentSession} />
              
              {/* Analysis Suggestions Button */}
              <div className="bg-gradient-to-r from-green-50 to-teal-50 p-4 rounded-lg border border-green-200">
                <div className="flex items-center justify-between">
                  <div>
                    <h4 className="font-medium text-gray-800">Need Analysis Ideas?</h4>
                    <p className="text-sm text-gray-600">Get AI-powered statistical analysis suggestions</p>
                  </div>
                  <button
                    onClick={() => getAnalysisSuggestions()}
                    disabled={isLoading || !geminiApiKey}
                    className="bg-green-600 text-white px-4 py-2 rounded hover:bg-green-700 disabled:opacity-50 text-sm"
                  >
                    Get Suggestions
                  </button>
                </div>
              </div>
              
              {messages.map(message => (
                <MessageRenderer key={message.id} message={message} />
              ))}
              
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 p-3 rounded-lg">
                    <div className="flex items-center space-x-2">
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                      <span className="text-gray-600">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
            
            <div className="bg-white border-t border-gray-200 p-4">
              <div className="flex space-x-2">
                <input
                  type="text"
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about your data..."
                  className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isLoading || !newMessage.trim()}
                  className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  Send
                </button>
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center bg-gray-50">
            <div className="text-center">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              <h3 className="mt-2 text-sm font-medium text-gray-900">No session selected</h3>
              <p className="mt-1 text-sm text-gray-500">Upload a CSV file to start analyzing your data</p>
            </div>
          </div>
        )}
      </div>

      {/* Right Panel */}
      <div className={`${rightPanelOpen ? 'w-96' : 'w-12'} transition-all duration-300 bg-white shadow-lg flex flex-col`}>
        <div className="p-4 border-b border-gray-200 flex items-center justify-between">
          <h2 className={`font-semibold text-gray-800 ${rightPanelOpen ? 'block' : 'hidden'}`}>
            Results & Analysis
          </h2>
          <button
            onClick={() => setRightPanelOpen(!rightPanelOpen)}
            className="p-2 hover:bg-gray-100 rounded"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </button>
        </div>
        
        {rightPanelOpen && (
          <>
            {/* Tab Navigation */}
            <div className="flex border-b border-gray-200">
              <button
                onClick={() => setActiveResultTab('execution')}
                className={`flex-1 py-2 px-4 text-sm font-medium ${
                  activeResultTab === 'execution'
                    ? 'border-b-2 border-blue-500 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Code Results
              </button>
              <button
                onClick={() => setActiveResultTab('history')}
                className={`flex-1 py-2 px-4 text-sm font-medium ${
                  activeResultTab === 'history'
                    ? 'border-b-2 border-blue-500 text-blue-600'
                    : 'text-gray-500 hover:text-gray-700'
                }`}
              >
                Analysis History
              </button>
            </div>
            
            <div className="flex-1 overflow-hidden">
              {activeResultTab === 'execution' ? <ExecutionPanel /> : <AnalysisHistoryPanel />}
            </div>
          </>
        )}
      </div>
      
      {showApiKeyModal && <ApiKeyModal />}
    </div>
  );
};

export default App;