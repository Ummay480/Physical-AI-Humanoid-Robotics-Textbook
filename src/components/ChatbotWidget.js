import React, { useState, useEffect, useRef } from 'react';

const ChatbotWidget = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { id: 1, text: 'Hello! I\'m your AI assistant for Physical AI & Humanoid Robotics. How can I help you today?', sender: 'bot' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const sendMessage = async () => {
    if (!inputValue.trim()) return;

    const userMessage = {
      id: Date.now(),
      text: inputValue,
      sender: 'user'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      // Simulate API call to RAG backend
      // In a real implementation, this would call your actual RAG API
      setTimeout(() => {
        const botResponse = {
          id: Date.now() + 1,
          text: getBotResponse(inputValue),
          sender: 'bot'
        };
        setMessages(prev => [...prev, botResponse]);
        setIsLoading(false);
      }, 1000);
    } catch (error) {
      const errorMessage = {
        id: Date.now() + 1,
        text: 'Sorry, I encountered an error processing your request.',
        sender: 'bot'
      };
      setMessages(prev => [...prev, errorMessage]);
      setIsLoading(false);
    }
  };

  const getBotResponse = (userInput) => {
    const input = userInput.toLowerCase();

    if (input.includes('hello') || input.includes('hi')) {
      return "Hello! I'm here to help you learn about Physical AI and Humanoid Robotics. What would you like to know?";
    } else if (input.includes('ros') || input.includes('robot operating system')) {
      return "ROS 2 (Robot Operating System 2) is a flexible framework for writing robot software. It's a collection of tools, libraries, and conventions that aim to simplify the task of creating complex and robust robot behavior across a wide variety of robot platforms.";
    } else if (input.includes('gazebo') || input.includes('simulation')) {
      return "Gazebo is a robot simulation environment that provides realistic physics, high-quality graphics, and convenient programmatic interfaces. It's commonly used for testing and validating robot algorithms before deployment on real hardware.";
    } else if (input.includes('urdf') || input.includes('model')) {
      return "URDF (Unified Robot Description Format) is an XML format used to model robots in ROS. It defines the robot's physical and visual properties including links, joints, and inertial parameters.";
    } else if (input.includes('physical ai') || input.includes('embodied intelligence')) {
      return "Physical AI refers to AI systems that function in the real physical world and comprehend physical laws. Embodied Intelligence emphasizes the importance of physical interaction with the environment for intelligent behavior.";
    } else {
      return "That's an interesting question about Physical AI and Humanoid Robotics! For more detailed information, I recommend checking the relevant chapters in the textbook. Is there a specific aspect of humanoid robotics you'd like to explore?";
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="chatbot-container">
      {isOpen ? (
        <div className="chatbot-window">
          <div className="chatbot-header">
            <strong>AI Assistant</strong>
            <button
              onClick={() => setIsOpen(false)}
              className="button button--sm button--danger"
            >
              ×
            </button>
          </div>
          <div className="chatbot-messages">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`margin-bottom--sm ${message.sender === 'user' ? 'text--right' : ''}`}
              >
                <div
                  className={`alert ${message.sender === 'user' ? 'alert--primary' : 'alert--secondary'}`}
                  style={{ display: 'inline-block', maxWidth: '80%' }}
                >
                  {message.text}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="margin-bottom--sm text--left">
                <div className="alert alert--secondary" style={{ display: 'inline-block', maxWidth: '80%' }}>
                  <em>Thinking...</em>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="chatbot-input">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about Physical AI..."
              disabled={isLoading}
              style={{ flex: 1, padding: '8px', marginRight: '8px' }}
            />
            <button
              onClick={sendMessage}
              disabled={isLoading || !inputValue.trim()}
              className="button button--primary"
            >
              Send
            </button>
          </div>
        </div>
      ) : (
        <button
          className="chatbot-button"
          onClick={() => setIsOpen(true)}
          aria-label="Open chatbot"
        >
          💬
        </button>
      )}
    </div>
  );
};

export default ChatbotWidget;