import React, { useState, useEffect } from 'react';

const TranslationButton = ({ contentKey, defaultText, urduText }) => {
  const [isUrdu, setIsUrdu] = useState(false);
  const [isAvailable, setIsAvailable] = useState(false);

  // Check if Urdu translation is available
  useEffect(() => {
    if (urduText && urduText.trim() !== '') {
      setIsAvailable(true);
    }
  }, [urduText]);

  const toggleLanguage = () => {
    if (!isAvailable) return;
    setIsUrdu(!isUrdu);
  };

  if (!isAvailable) {
    return <span>{defaultText}</span>;
  }

  return (
    <div className="translation-component">
      <button
        onClick={toggleLanguage}
        className="button button--sm button--outline"
        style={{ marginBottom: '1rem' }}
      >
        {isUrdu ? 'English' : 'اردو میں تبدیل کریں'}
      </button>
      <div>
        {isUrdu ? urduText : defaultText}
      </div>
    </div>
  );
};

export default TranslationButton;