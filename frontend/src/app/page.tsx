'use client';

import { useState, useRef, useEffect } from 'react';

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [clientId, setClientId] = useState<number>(1);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    const assistantMessage: Message = {
      id: (Date.now() + 1).toString(),
      type: 'assistant',
      content: '',
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, assistantMessage]);

    try {
      const response = await fetch('http://localhost:3001/ask', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: userMessage.content, clientId }),
      });

      if (!response.body) {
        throw new Error('No response body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let accumulatedContent = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        
        if (!chunk.includes('---SOURCES---')) {
          accumulatedContent += chunk;
          setMessages(prev => 
            prev.map(msg => 
              msg.id === assistantMessage.id 
                ? { ...msg, content: accumulatedContent }
                : msg
            )
          );
        }
      }
    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => 
        prev.map(msg => 
          msg.id === assistantMessage.id 
            ? { ...msg, content: 'Sorry, there was an error processing your request.' }
            : msg
        )
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 flex flex-col relative overflow-hidden">
    <div className="absolute inset-0 overflow-hidden pointer-events-none">
      <div className="absolute -top-40 -right-40 w-80 h-80 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-pulse"></div>
      <div className="absolute -bottom-40 -left-40 w-80 h-80 bg-indigo-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-pulse animation-delay-2"></div>
    </div>

    {/* Header */}
    <div className="relative bg-white/80 backdrop-blur-md shadow-lg border-b border-white/20 px-6 py-6">
      <div className="flex justify-between items-center max-w-4xl mx-auto">
        <div className="flex items-center space-x-4">
          <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
            <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
          </div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-800 to-gray-600 bg-clip-text text-transparent">
              ASK AI
            </h1>
            <p className="text-gray-600 text-sm font-medium">AI-powered responses with intelligent sourcing</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          <label htmlFor="clientId" className="text-sm font-semibold text-gray-700">Client:</label>
          <div className="relative">
            <select
              id="clientId"
              value={clientId}
              onChange={(e) => setClientId(Number(e.target.value))}
              className="appearance-none bg-white/80 backdrop-blur-sm px-4 py-2 pr-8 border border-gray-200 rounded-xl text-sm font-medium focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none shadow-sm hover:shadow-md transition-all duration-200"
            >
              <option value={1}>Nathan (ID: 1)</option>
              <option value={2}>Robert (ID: 2)</option>
            </select>
            <div className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
              <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </div>
      </div>
    </div>

    {/* Main Chat Area */}
    <div className="flex-1 flex flex-col max-w-4xl mx-auto w-full px-4 py-6 relative z-10">
      <div className="flex-1 bg-white/70 backdrop-blur-sm rounded-2xl shadow-2xl border border-white/30 mb-6 flex flex-col overflow-hidden">
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {messages.length === 0 && (
            <div className="text-center text-gray-500 py-20">
              <div className="relative mb-6">
                <div className="text-7xl animate-bounce">💬</div>
                <div className="absolute inset-0 text-7xl animate-ping opacity-20">💬</div>
              </div>
              <h3 className="text-xl font-semibold mb-3 text-gray-700">Start Your Conversation</h3>
              <p className="text-sm text-gray-500 max-w-md mx-auto leading-relaxed">
                Ask any question and I'll provide intelligent responses backed by relevant sources and context.
              </p>
            </div>
          )}
          
          {messages.map((message, index) => (
            <div
              key={message.id}
              className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'} animate-fade-in-up`}
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              <div className={`flex items-start space-x-3 max-w-xs lg:max-w-2xl ${message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''}`}>
                {/* Avatar */}
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center shadow-md ${
                  message.type === 'user' 
                    ? 'bg-gradient-to-br from-blue-500 to-blue-600' 
                    : 'bg-gradient-to-br from-gray-500 to-gray-600'
                }`}>
                  {message.type === 'user' ? (
                    <svg className="w-4 h-4 text-white" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M10 9a3 3 0 100-6 3 3 0 000 6zm-7 9a7 7 0 1114 0H3z" clipRule="evenodd" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                    </svg>
                  )}
                </div>
                
                {/* Message Bubble */}
                <div className={`px-4 py-3 rounded-2xl shadow-md hover:shadow-lg transition-shadow duration-200 ${
                  message.type === 'user'
                    ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white rounded-br-md'
                    : 'bg-white border border-gray-100 text-gray-800 rounded-bl-md'
                }`}>
                  <div className="whitespace-pre-wrap text-sm leading-relaxed">
                    {message.content || (isLoading && message.type === 'assistant' && (
                      <div className="flex items-center space-x-2">
                        <div className="flex space-x-1">
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                          <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                        </div>
                        <span className="text-gray-500 text-xs">Thinking...</span>
                      </div>
                    ))}
                  </div>
                  <div className={`text-xs mt-2 ${message.type === 'user' ? 'text-blue-100' : 'text-gray-400'}`}>
                    {new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </div>
                </div>
              </div>
            </div>
          ))}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Form */}
      <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-white/30 p-6">
        <div className="flex space-x-4">
          <div className="flex-1 relative">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  handleSubmit(e);
                }
              }}
              placeholder="Ask me anything..."
              className="w-full px-6 py-4 bg-white/80 backdrop-blur-sm border border-gray-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none text-gray-800 placeholder-gray-400 text-sm font-medium shadow-sm hover:shadow-md transition-all duration-200"
              disabled={isLoading}
            />
            {input && (
              <button
                type="button"
                onClick={() => setInput('')}
                className="absolute right-3 top-1/2 transform -translate-y-1/2 p-1 text-gray-400 hover:text-gray-600 transition-colors"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
          <button
            onClick={handleSubmit}
            disabled={isLoading || !input.trim()}
            className="px-8 py-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl hover:from-blue-600 hover:to-blue-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 font-semibold text-sm shadow-lg hover:shadow-xl transform hover:scale-105 disabled:transform-none flex items-center space-x-2"
          >
            {isLoading ? (
              <>
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin"></div>
                <span>Sending...</span>
              </>
            ) : (
              <>
                <span>Send</span>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </svg>
              </>
            )}
          </button>
        </div>
        
        {/* Quick Actions */}
        <div className="flex flex-wrap gap-2 mt-4"> 
          {['Suggest some plan to him?', 'List down all the principles of effective intervention?', 'Is he going out on 2nd June?'].map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              onClick={() => setInput(suggestion)}
              className="px-3 py-1.5 text-xs bg-gray-100 hover:bg-gray-200 text-gray-600 rounded-lg transition-colors duration-200 border border-gray-200 hover:border-gray-300"
              disabled={isLoading}
            >
              {suggestion}
            </button>
          ))}
        </div>
      </div>
    </div>

    <style jsx>{`
      @keyframes fade-in-up {
        from {
          opacity: 0;
          transform: translateY(20px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      
      .animate-fade-in-up {
        animation: fade-in-up 0.5s ease-out forwards;
      }
      
      .animation-delay-2 {
        animation-delay: 2s;
      }
      
      /* Custom scrollbar */
      .overflow-y-auto::-webkit-scrollbar {
        width: 6px;
      }
      
      .overflow-y-auto::-webkit-scrollbar-track {
        background: transparent;
      }
      
      .overflow-y-auto::-webkit-scrollbar-thumb {
        background: rgba(156, 163, 175, 0.3);
        border-radius: 3px;
      }
      
      .overflow-y-auto::-webkit-scrollbar-thumb:hover {
        background: rgba(156, 163, 175, 0.5);
      }
    `}</style>
  </div>
  );
}