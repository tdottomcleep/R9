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
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  
  const fileInputRef = useRef(null);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    fetchSessions();
  }, []);

  useEffect(() => {
    if (currentSession) {
      fetchMessages(currentSession.id);
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

  const handleExecuteCode = async (code) => {
    if (!currentSession || !geminiApiKey) return;

    setIsLoading(true);
    try {
      const response = await axios.post(`${API}/sessions/${currentSession.id}/execute`, {
        session_id: currentSession.id,
        code: code,
        gemini_api_key: geminiApiKey
      });

      setExecutionResult(response.data);
      setRightPanelOpen(true);
    } catch (error) {
      console.error('Error executing code:', error);
      setExecutionResult({
        success: false,
        error: 'Error executing code: ' + error.message,
        output: '',
        plots: []
      });
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

    const { columns, shape, head, dtypes, null_counts } = session.csv_preview;

    return (
      <div className="bg-gray-50 p-4 rounded-lg mb-4">
        <h3 className="font-semibold text-gray-700 mb-2">Data Overview</h3>
        <div className="text-sm text-gray-600 space-y-1">
          <p><strong>Shape:</strong> {shape?.[0]} rows × {shape?.[1]} columns</p>
          <p><strong>Columns:</strong> {columns?.join(', ')}</p>
        </div>
        
        {head && head.length > 0 && (
          <div className="mt-3">
            <h4 className="font-medium text-gray-700 mb-2">Sample Data:</h4>
            <div className="overflow-x-auto">
              <table className="min-w-full text-xs">
                <thead>
                  <tr className="bg-gray-100">
                    {columns?.map(col => (
                      <th key={col} className="px-2 py-1 text-left">{col}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {head.slice(0, 3).map((row, idx) => (
                    <tr key={idx} className="border-b">
                      {columns?.map(col => (
                        <td key={col} className="px-2 py-1">{row[col]}</td>
                      ))}
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
              <img 
                key={index} 
                src={`data:image/png;base64,${plot}`} 
                alt={`Plot ${index + 1}`}
                className="max-w-full rounded shadow-lg mb-2"
              />
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
            Analysis Results
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
          <div className="flex-1 overflow-hidden">
            <ExecutionPanel />
          </div>
        )}
      </div>
      
      {showApiKeyModal && <ApiKeyModal />}
    </div>
  );
};

export default App;