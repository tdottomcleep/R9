import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import axios from 'axios';

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

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
  
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    if (currentSession) {
      fetchMessages(currentSession.id);
      fetchAnalysisHistory(currentSession.id);
    }
  }, [currentSession]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const fetchSessions = async () => {
    try {
      const response = await axios.get(`${API}/sessions`);
      setSessions(response.data);
    } catch (error) {
      console.error('Error fetching sessions:', error);
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

      const response = await axios.post(`${API}/sessions`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      await fetchSessions();
      setCurrentSession(response.data);
      setMessages([]);
      setExecutionResult(null);
    } catch (error) {
      console.error('Error uploading file:', error);
      alert('Error uploading file. Please try again.');
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
                  return <CodeBlock key={index} code={part} onExecute={handleExecuteCode} />;
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
        <h3 className="font-semibold text-gray-700 border-b pb-2">Execution Results</h3>
        
        {executionResult.success ? (
          <div className="text-green-600 text-sm">✅ Execution successful</div>
        ) : (
          <div className="text-red-600 text-sm">❌ Execution failed</div>
        )}
        
        {executionResult.output && (
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Output:</h4>
            <pre className="bg-gray-100 p-3 rounded text-sm overflow-x-auto">
              {executionResult.output}
            </pre>
          </div>
        )}
        
        {executionResult.error && (
          <div>
            <h4 className="font-medium text-red-700 mb-2">Error:</h4>
            <pre className="bg-red-50 p-3 rounded text-sm overflow-x-auto text-red-700">
              {executionResult.error}
            </pre>
          </div>
        )}
        
        {executionResult.plots && executionResult.plots.length > 0 && (
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Visualizations:</h4>
            {executionResult.plots.map((plot, index) => (
              <div key={index} className="mb-4">
                {plot.type === 'matplotlib' ? (
                  <img 
                    src={`data:image/png;base64,${plot.data}`} 
                    alt={`Plot ${index + 1}`}
                    className="max-w-full rounded shadow-lg"
                  />
                ) : plot.type === 'plotly' ? (
                  <div 
                    dangerouslySetInnerHTML={{ __html: plot.html }}
                    className="plotly-container"
                  />
                ) : (
                  <img 
                    src={`data:image/png;base64,${plot}`} 
                    alt={`Plot ${index + 1}`}
                    className="max-w-full rounded shadow-lg"
                  />
                )}
              </div>
            ))}
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
      <div className="flex-1 flex flex-col">
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