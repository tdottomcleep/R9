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
      
      const leftPanelWidth = leftPanelOpen ? 320 : 48; // 80*4 = 320px, 12*4 = 48px
      const containerWidth = window.innerWidth - leftPanelWidth;
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

  // Table Parser and Renderer Component
  const TableRenderer = ({ content, title }) => {
    const parseTableContent = (content) => {
      const lines = content.split('\n').filter(line => line.trim());
      
      // Try to detect if it's a pandas DataFrame output
      if (lines.length < 2) return null;
      
      // Look for pandas-style table (with index and columns)
      const rows = [];
      let headers = [];
      
      // Find the header row (usually the first non-empty line)
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();
        if (line && !line.startsWith('-') && !line.startsWith('=')) {
          // Check if this looks like a data row
          const parts = line.split(/\s{2,}|\t/).filter(p => p.trim());
          if (parts.length > 1) {
            if (headers.length === 0) {
              // First data row - extract headers
              headers = parts;
            } else {
              // Data row
              rows.push(parts);
            }
          }
        }
      }
      
      // If we didn't find proper headers, try a different approach
      if (headers.length === 0) {
        // Try to parse as pipe-separated or whitespace-separated
        const tableLines = lines.filter(line => 
          line.includes('|') || 
          (line.trim() && !line.startsWith('-') && !line.startsWith('='))
        );
        
        if (tableLines.length > 0) {
          const firstLine = tableLines[0];
          if (firstLine.includes('|')) {
            // Pipe-separated table
            headers = firstLine.split('|').map(h => h.trim()).filter(h => h);
            for (let i = 1; i < tableLines.length; i++) {
              const rowData = tableLines[i].split('|').map(d => d.trim()).filter(d => d);
              if (rowData.length > 0) {
                rows.push(rowData);
              }
            }
          } else {
            // Whitespace-separated table
            headers = firstLine.split(/\s{2,}/).map(h => h.trim()).filter(h => h);
            for (let i = 1; i < tableLines.length; i++) {
              const rowData = tableLines[i].split(/\s{2,}/).map(d => d.trim()).filter(d => d);
              if (rowData.length > 0) {
                rows.push(rowData);
              }
            }
          }
        }
      }
      
      return { headers, rows };
    };

    const tableData = parseTableContent(content);
    
    if (!tableData || tableData.headers.length === 0) {
      // Fallback to pre-formatted text if parsing fails
      return (
        <pre className="text-sm text-gray-800 whitespace-pre-wrap font-mono">
          {content}
        </pre>
      );
    }
    
    return (
      <div className="overflow-x-auto">
        <table className="min-w-full border border-gray-200">
          <thead className="bg-gray-50">
            <tr>
              {tableData.headers.map((header, index) => (
                <th key={index} className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase tracking-wider border-b border-gray-200">
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="bg-white divide-y divide-gray-200">
            {tableData.rows.map((row, rowIndex) => (
              <tr key={rowIndex} className="hover:bg-gray-50">
                {row.map((cell, cellIndex) => (
                  <td key={cellIndex} className="px-3 py-2 text-sm text-gray-900 border-b border-gray-200">
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
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

  // Enhanced Code Block Component
  const JuliusStyleCodeBlock = ({ code, onExecute, sectionTitle = "Python Code" }) => {
    const [isExpanded, setIsExpanded] = useState(false);
    const [showLineNumbers, setShowLineNumbers] = useState(true);
    
    const codeLines = code.split('\n');
    const displayLines = isExpanded ? codeLines : codeLines.slice(0, 8);
    const hasMoreLines = codeLines.length > 8;
    
    return (
      <div className="bg-slate-50 border border-slate-200 rounded-xl my-4 shadow-sm overflow-hidden">
        {/* Header */}
        <div className="bg-gradient-to-r from-slate-100 to-slate-50 border-b border-slate-200 px-5 py-4 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-2 h-2 bg-emerald-500 rounded-full"></div>
            <h4 className="text-sm font-semibold text-slate-700">{sectionTitle}</h4>
            <span className="text-xs text-slate-500 bg-slate-200 px-2 py-1 rounded-full">
              {codeLines.length} lines
            </span>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowLineNumbers(!showLineNumbers)}
              className="text-xs text-slate-500 hover:text-slate-700 px-3 py-1.5 rounded-full hover:bg-slate-200 transition-all duration-200"
            >
              {showLineNumbers ? 'Hide' : 'Show'} Lines
            </button>
            <button
              onClick={() => onExecute(code)}
              className="bg-emerald-600 text-white px-4 py-1.5 rounded-full text-xs hover:bg-emerald-700 transition-all duration-200 shadow-sm hover:shadow-md"
            >
              ▶ Execute
            </button>
          </div>
        </div>
        
        {/* Code Content */}
        <div className="relative">
          <div 
            className="overflow-auto text-sm bg-slate-50"
            style={{ maxHeight: '350px' }}
          >
            <div className="flex">
              {/* Line Numbers */}
              {showLineNumbers && (
                <div className="bg-slate-100 text-slate-500 px-3 py-3 border-r border-slate-200 select-none">
                  {displayLines.map((_, index) => (
                    <div key={index} className="text-right font-mono text-xs leading-6">
                      {index + 1}
                    </div>
                  ))}
                </div>
              )}
              
              {/* Code */}
              <div className="flex-1 px-4 py-3">
                <pre className="text-slate-800 font-mono text-xs leading-6 whitespace-pre-wrap">
                  {displayLines.join('\n')}
                </pre>
              </div>
            </div>
          </div>
          
          {/* Expand/Collapse Button */}
          {hasMoreLines && (
            <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-slate-50 to-transparent p-3 text-center">
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="text-emerald-600 hover:text-emerald-800 text-sm font-medium bg-white border border-slate-300 px-4 py-2 rounded-full shadow-sm hover:bg-slate-50 transition-all duration-200"
              >
                {isExpanded ? 'Show Less' : `Show ${codeLines.length - 8} More Lines`}
              </button>
            </div>
          )}
        </div>
      </div>
    );
  };

  // Enhanced Analysis Block Component
  const AnalysisBlock = ({ title, children, icon = "📊" }) => {
    return (
      <div className="bg-blue-50 border border-blue-200 rounded-xl my-4 shadow-sm overflow-hidden">
        <div className="bg-gradient-to-r from-blue-100 to-blue-50 border-b border-blue-200 px-5 py-3">
          <div className="flex items-center space-x-3">
            <span className="text-sm">{icon}</span>
            <h4 className="text-sm font-semibold text-blue-800">{title}</h4>
          </div>
        </div>
        <div className="px-5 py-4 text-slate-700">
          {children}
        </div>
      </div>
    );
  };

  // Enhanced Suggestion Block Component
  const SuggestionBlock = ({ title, children, icon = "💡" }) => {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-xl my-4 shadow-sm overflow-hidden">
        <div className="bg-gradient-to-r from-amber-100 to-amber-50 border-b border-amber-200 px-5 py-3">
          <div className="flex items-center space-x-3">
            <span className="text-sm">{icon}</span>
            <h4 className="text-sm font-semibold text-amber-800">{title}</h4>
          </div>
        </div>
        <div className="px-5 py-4 text-slate-700">
          {children}
        </div>
      </div>
    );
  };

  // Clickable Analysis Button Component
  const AnalysisButton = ({ text, onClick, type = "primary" }) => {
    const baseClasses = "inline-flex items-center px-3 py-1.5 rounded-full text-sm font-medium transition-all duration-200 cursor-pointer hover:shadow-md";
    const typeClasses = {
      primary: "bg-blue-600 text-white hover:bg-blue-700",
      secondary: "bg-slate-200 text-slate-700 hover:bg-slate-300",
      success: "bg-emerald-600 text-white hover:bg-emerald-700",
      warning: "bg-amber-600 text-white hover:bg-amber-700"
    };
    
    return (
      <button
        onClick={onClick}
        className={`${baseClasses} ${typeClasses[type]}`}
      >
        {text}
      </button>
    );
  };

  const CodeBlock = ({ code, onExecute }) => {
    return (
      <div className="bg-slate-900 border border-slate-700 rounded-xl my-3 shadow-sm overflow-hidden">
        <div className="flex justify-between items-center p-3 bg-slate-800 border-b border-slate-700">
          <span className="text-slate-300 text-xs font-medium">Code</span>
          <button
            onClick={() => onExecute(code)}
            className="bg-emerald-600 text-white px-3 py-1.5 rounded-full text-xs hover:bg-emerald-700 transition-all duration-200"
          >
            ▶ Execute
          </button>
        </div>
        <div className="p-4">
          <pre className="text-emerald-400 text-sm overflow-x-auto font-mono leading-relaxed">
            {code}
          </pre>
        </div>
      </div>
    );
  };

  // Enhanced Message Parser and Renderer
  const parseAndRenderAIResponse = (content) => {
    // Clean up markdown formatting and bold important headings
    const cleanContent = content
      .replace(/\*\*\*(.*?)\*\*\*/g, '<strong>$1</strong>')
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/###\s*(.*?)(?=\n|$)/g, '<strong>$1</strong>')
      .replace(/##\s*(.*?)(?=\n|$)/g, '<strong>$1</strong>')
      .replace(/#{1,6}\s*(.*?)(?=\n|$)/g, '<strong>$1</strong>')
      .replace(/\*\s*/g, '• ')
      .replace(/(\d+\.\s*[A-Z][^.]*:)/g, '<strong>$1</strong>') // Bold numbered headings
      .replace(/([A-Z][^.]*:)(?=\n|$)/g, '<strong>$1</strong>'); // Bold section headings

    // Split content into sections
    const sections = [];
    let currentSection = { type: 'text', content: '' };
    
    const lines = cleanContent.split('\n');
    let inCodeBlock = false;
    let codeContent = '';
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      // Handle code blocks
      if (line.startsWith('```python')) {
        if (currentSection.content.trim()) {
          sections.push(currentSection);
        }
        inCodeBlock = true;
        codeContent = '';
        currentSection = { type: 'text', content: '' };
        continue;
      }
      
      if (line.startsWith('```') && inCodeBlock) {
        sections.push({ type: 'code', content: codeContent, title: 'Python Code' });
        inCodeBlock = false;
        codeContent = '';
        currentSection = { type: 'text', content: '' };
        continue;
      }
      
      if (inCodeBlock) {
        codeContent += line + '\n';
        continue;
      }
      
      // Detect analysis/suggestion patterns
      if (line.toLowerCase().includes('analysis') || 
          line.toLowerCase().includes('describe') ||
          line.toLowerCase().includes('summary') ||
          line.toLowerCase().includes('overview') ||
          line.toLowerCase().includes('statistics')) {
        if (currentSection.content.trim()) {
          sections.push(currentSection);
        }
        currentSection = { type: 'analysis', content: line + '\n', title: 'Data Analysis' };
        continue;
      }
      
      if (line.toLowerCase().includes('suggest') || 
          line.toLowerCase().includes('recommend') ||
          line.toLowerCase().includes('you can') ||
          line.toLowerCase().includes('would you like') ||
          line.toLowerCase().includes('potential') ||
          line.toLowerCase().includes('visualization')) {
        if (currentSection.content.trim()) {
          sections.push(currentSection);
        }
        currentSection = { type: 'suggestion', content: line + '\n', title: 'Suggestions' };
        continue;
      }
      
      currentSection.content += line + '\n';
    }
    
    if (currentSection.content.trim()) {
      sections.push(currentSection);
    }
    
    return sections;
  };

  // Enhanced suggestion text processor
  const processSuggestionText = (text) => {
    // Define statistical test patterns
    const testPatterns = [
      { pattern: /\b(paired t-test|paired t test)\b/gi, name: 'Paired T-Test' },
      { pattern: /\b(unpaired t-test|unpaired t test|independent t-test)\b/gi, name: 'Independent T-Test' },
      { pattern: /\b(anova|one-way anova|ANOVA)\b/gi, name: 'ANOVA' },
      { pattern: /\b(chi-square|chi square|χ²)\b/gi, name: 'Chi-Square Test' },
      { pattern: /\b(correlation|pearson correlation|spearman correlation)\b/gi, name: 'Correlation Analysis' },
      { pattern: /\b(regression|linear regression|logistic regression)\b/gi, name: 'Regression Analysis' },
      { pattern: /\b(mann-whitney|mann whitney|wilcoxon)\b/gi, name: 'Mann-Whitney Test' },
      { pattern: /\b(kruskal-wallis|kruskal wallis)\b/gi, name: 'Kruskal-Wallis Test' },
      { pattern: /\b(fisher's exact|fisher exact)\b/gi, name: "Fisher's Exact Test" },
      { pattern: /\b(two-way anova|two way anova)\b/gi, name: 'Two-Way ANOVA' }
    ];
    
    // Split text into parts and identify clickable elements
    const parts = [];
    let lastIndex = 0;
    let matches = [];
    
    // Find all matches
    testPatterns.forEach(({ pattern, name }) => {
      const regex = new RegExp(pattern.source, pattern.flags);
      let match;
      while ((match = regex.exec(text)) !== null) {
        matches.push({
          start: match.index,
          end: match.index + match[0].length,
          text: match[0],
          name: name
        });
      }
    });
    
    // Sort matches by position
    matches.sort((a, b) => a.start - b.start);
    
    // Build parts array
    matches.forEach(match => {
      // Add text before match
      if (match.start > lastIndex) {
        parts.push({
          type: 'text',
          content: text.slice(lastIndex, match.start)
        });
      }
      
      // Add clickable match
      parts.push({
        type: 'button',
        content: match.text,
        name: match.name,
        action: () => handleAnalysisClick(match.name)
      });
      
      lastIndex = match.end;
    });
    
    // Add remaining text
    if (lastIndex < text.length) {
      parts.push({
        type: 'text',
        content: text.slice(lastIndex)
      });
    }
    
    return parts.length > 0 ? parts : [{ type: 'text', content: text }];
  };

  // Handle analysis button clicks
  const handleAnalysisClick = (analysisType) => {
    const analysisTemplates = {
      'Paired T-Test': `# Paired T-Test Analysis
# Comparing measurements from the same subjects

import scipy.stats as stats
import pandas as pd

# Assuming your data has 'before' and 'after' columns
# Replace with your actual column names
before = df['before_measurement']
after = df['after_measurement']

# Perform paired t-test
t_stat, p_value = stats.ttest_rel(before, after)

print(f"Paired T-Test Results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("Result: Significant difference (p < 0.05)")
else:
    print("Result: No significant difference (p >= 0.05)")`,
      
      'Independent T-Test': `# Independent T-Test Analysis
# Comparing two independent groups

import scipy.stats as stats
import pandas as pd

# Assuming your data has a grouping variable
# Replace with your actual column names
group1 = df[df['group'] == 'Group1']['measurement']
group2 = df[df['group'] == 'Group2']['measurement']

# Perform independent t-test
t_stat, p_value = stats.ttest_ind(group1, group2)

print(f"Independent T-Test Results:")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("Result: Significant difference between groups (p < 0.05)")
else:
    print("Result: No significant difference between groups (p >= 0.05)")`,
      
      'ANOVA': `# One-Way ANOVA Analysis
# Comparing means across multiple groups

import scipy.stats as stats
import pandas as pd

# Assuming your data has a grouping variable
# Replace with your actual column names
groups = []
for group_name in df['group'].unique():
    groups.append(df[df['group'] == group_name]['measurement'])

# Perform one-way ANOVA
f_stat, p_value = stats.f_oneway(*groups)

print(f"One-Way ANOVA Results:")
print(f"F-statistic: {f_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpretation
if p_value < 0.05:
    print("Result: Significant difference between groups (p < 0.05)")
    print("Post-hoc tests recommended to identify specific group differences")
else:
    print("Result: No significant difference between groups (p >= 0.05)")`,
      
      'Chi-Square Test': `# Chi-Square Test Analysis
# Testing independence between categorical variables

import scipy.stats as stats
import pandas as pd

# Create contingency table
# Replace with your actual column names
contingency_table = pd.crosstab(df['variable1'], df['variable2'])

print("Contingency Table:")
print(contingency_table)

# Perform chi-square test
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

print(f"\\nChi-Square Test Results:")
print(f"Chi-square statistic: {chi2:.4f}")
print(f"P-value: {p_value:.4f}")
print(f"Degrees of freedom: {dof}")

# Interpretation
if p_value < 0.05:
    print("Result: Significant association between variables (p < 0.05)")
else:
    print("Result: No significant association between variables (p >= 0.05)")`,
      
      'Correlation Analysis': `# Correlation Analysis
# Examining relationships between continuous variables

import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt

# Replace with your actual column names
var1 = df['variable1']
var2 = df['variable2']

# Calculate Pearson correlation
pearson_r, pearson_p = stats.pearsonr(var1, var2)

# Calculate Spearman correlation
spearman_r, spearman_p = stats.spearmanr(var1, var2)

print(f"Correlation Analysis Results:")
print(f"Pearson correlation: r = {pearson_r:.4f}, p = {pearson_p:.4f}")
print(f"Spearman correlation: ρ = {spearman_r:.4f}, p = {spearman_p:.4f}")

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(var1, var2, alpha=0.6)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.title('Scatter Plot with Correlation')
plt.show()

# Interpretation
if pearson_p < 0.05:
    print("Result: Significant correlation detected (p < 0.05)")
else:
    print("Result: No significant correlation (p >= 0.05)")`
    };
    
    const code = analysisTemplates[analysisType];
    if (code) {
      handleExecuteCode(code);
    }
  };

  const MessageRenderer = ({ message }) => {
    const isUser = message.role === 'user';
    const content = message.content;
    
    if (isUser) {
      return (
        <div className="flex justify-end mb-4">
          <div className="max-w-3xl bg-blue-600 text-white p-4 rounded-2xl shadow-sm">
            <p className="whitespace-pre-wrap text-sm leading-relaxed">{content}</p>
          </div>
        </div>
      );
    }
    
    // Parse AI response into sections
    const sections = parseAndRenderAIResponse(content);
    
    return (
      <div className="flex justify-start mb-6">
        <div className="max-w-4xl w-full space-y-3">
          {sections.map((section, index) => {
            switch (section.type) {
              case 'code':
                return (
                  <JuliusStyleCodeBlock 
                    key={index} 
                    code={section.content.trim()} 
                    onExecute={handleExecuteCode} 
                    sectionTitle={section.title}
                  />
                );
              
              case 'analysis':
                return (
                  <AnalysisBlock key={index} title={section.title} icon="📊">
                    <div className="prose prose-sm max-w-none">
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">
                        {section.content.trim()}
                      </p>
                    </div>
                  </AnalysisBlock>
                );
              
              case 'suggestion':
                const suggestionParts = processSuggestionText(section.content);
                return (
                  <SuggestionBlock key={index} title={section.title} icon="💡">
                    <div className="prose prose-sm max-w-none">
                      <div className="text-sm leading-relaxed">
                        {suggestionParts.map((part, partIndex) => {
                          if (part.type === 'button') {
                            return (
                              <AnalysisButton
                                key={partIndex}
                                text={part.content}
                                onClick={part.action}
                                type="primary"
                              />
                            );
                          }
                          return (
                            <span key={partIndex} className="whitespace-pre-wrap">
                              {part.content}
                            </span>
                          );
                        })}
                      </div>
                    </div>
                  </SuggestionBlock>
                );
              
              default:
                return (
                  <div key={index} className="bg-slate-50 border border-slate-200 rounded-xl p-4 shadow-sm">
                    <div className="prose prose-sm max-w-none">
                      <div 
                        className="whitespace-pre-wrap text-sm leading-relaxed text-slate-700"
                        dangerouslySetInnerHTML={{ __html: section.content.trim() }}
                      />
                    </div>
                  </div>
                );
            }
          })}
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
                      <h5 className="text-sm font-medium text-gray-700 mb-2">Data Tables</h5>
                      <div className="space-y-4">
                        {section.tables.map((table, tableIndex) => {
                          // Filter out redundant or confusing table titles
                          const tableTitle = table.title;
                          const shouldShowTable = tableTitle && 
                            !tableTitle.toLowerCase().includes(': df') &&
                            !tableTitle.toLowerCase().includes('survival analysis: df') &&
                            tableTitle !== 'df';
                          
                          return shouldShowTable ? (
                            <div key={tableIndex} className="bg-white border border-gray-200 rounded-lg overflow-hidden">
                              <div className="bg-gray-50 px-3 py-2 border-b border-gray-200">
                                <span className="text-xs font-medium text-gray-600">
                                  {table.title.replace(/:\s*df$/i, '').trim()}
                                </span>
                              </div>
                              <div className="p-3 overflow-auto">
                                <TableRenderer content={table.content} title={table.title} />
                              </div>
                            </div>
                          ) : null;
                        })}
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
        className="flex flex-col bg-white transition-all duration-200"
        style={{ 
          flex: rightPanelOpen ? `0 0 ${centerPanelWidth}%` : '1'
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
                <div className="flex justify-start mb-4">
                  <div className="bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-xl p-4 shadow-sm">
                    <div className="flex items-center space-x-3">
                      <div className="animate-spin rounded-full h-5 w-5 border-2 border-blue-600 border-t-transparent"></div>
                      <span className="text-blue-700 font-medium">AI is analyzing...</span>
                    </div>
                  </div>
                </div>
              )}
              
              <div ref={messagesEndRef} />
            </div>
            
            <div className="bg-white border-t border-gray-200 p-4">
              <div className="flex space-x-3">
                <textarea
                  value={newMessage}
                  onChange={(e) => setNewMessage(e.target.value)}
                  onKeyPress={handleKeyPress}
                  placeholder="Ask about your data..."
                  className="flex-1 px-4 py-3 border border-gray-300 rounded-xl focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                  rows="2"
                />
                <button
                  onClick={handleSendMessage}
                  disabled={isLoading || !newMessage.trim()}
                  className="bg-blue-600 text-white px-4 py-3 rounded-xl hover:bg-blue-700 disabled:opacity-50 transition-all duration-200 flex items-center space-x-2"
                >
                  <span>Send</span>
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                  </svg>
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

      {/* Draggable Divider */}
      {rightPanelOpen && (
        <div
          className="w-1 bg-gray-300 hover:bg-blue-500 cursor-col-resize transition-colors duration-200 relative group"
          onMouseDown={handleDragStart}
        >
          <div className="absolute inset-y-0 -left-1 -right-1 group-hover:bg-blue-500 group-hover:bg-opacity-20"></div>
        </div>
      )}

      {/* Right Panel */}
      <div 
        className={`${rightPanelOpen ? 'flex-1' : 'w-12'} transition-all duration-300 bg-white shadow-lg flex flex-col`}
        style={{ 
          width: rightPanelOpen ? `${100 - centerPanelWidth}%` : '48px'
        }}
      >
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