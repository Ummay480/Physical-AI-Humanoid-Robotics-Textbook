import React, { useState, useEffect } from 'react';
import { useAuth } from './AuthContext';

const ContentPersonalization = ({ children, contentKey }) => {
  const { user } = useAuth();
  const [personalizedContent, setPersonalizedContent] = useState(null);
  const [showPersonalizeButton, setShowPersonalizeButton] = useState(false);

  useEffect(() => {
    // Check if user is logged in and has background info
    if (user && user.background) {
      setShowPersonalizeButton(true);
    } else {
      setShowPersonalizeButton(false);
    }
  }, [user]);

  const personalizeContent = () => {
    if (!user || !user.background) {
      alert('Please register and provide your background information to enable content personalization.');
      return;
    }

    // In a real implementation, this would fetch personalized content based on user background
    // For demo purposes, we'll just show a message
    const background = user.background.toLowerCase();
    let level = 'beginner';

    if (background.includes('expert') || background.includes('senior') || background.includes('advanced')) {
      level = 'advanced';
    } else if (background.includes('intermediate') || background.includes('mid')) {
      level = 'intermediate';
    }

    // In a real implementation, we would fetch different content based on user level
    setPersonalizedContent(
      <div className="alert alert--info">
        <p><strong>Personalized Content:</strong> Content adapted for {level} level based on your background: {user.background}</p>
        <p>This section would show content tailored to your experience level.</p>
      </div>
    );
  };

  return (
    <div className="content-personalization">
      {showPersonalizeButton && (
        <button
          className="button button--primary button--sm"
          onClick={personalizeContent}
          style={{ marginBottom: '1rem' }}
        >
          Personalize Content
        </button>
      )}

      {personalizedContent && (
        <div className="personalized-content">
          {personalizedContent}
        </div>
      )}

      <div className="original-content">
        {children}
      </div>
    </div>
  );
};

export default ContentPersonalization;