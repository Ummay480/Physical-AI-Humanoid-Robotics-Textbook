import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check if user is logged in on component mount
    const token = localStorage.getItem('auth-token');
    if (token) {
      // In a real app, you would verify the token with your backend
      try {
        // Decode token and set user
        const userData = JSON.parse(atob(token.split('.')[1]));
        setUser(userData);
      } catch (error) {
        console.error('Error decoding token:', error);
        localStorage.removeItem('auth-token');
      }
    }
    setLoading(false);
  }, []);

  const login = async (email, password) => {
    // In a real app, you would call your authentication API
    // For demo purposes, we'll simulate a successful login
    const userData = {
      id: 1,
      email: email,
      name: email.split('@')[0],
      role: email === 'admin@example.com' ? 'admin' : 'user'
    };

    // Create a mock token (in a real app, this would come from your backend)
    const token = btoa(JSON.stringify(userData));
    localStorage.setItem('auth-token', token);
    setUser(userData);
    return userData;
  };

  const logout = () => {
    localStorage.removeItem('auth-token');
    setUser(null);
  };

  const value = {
    user,
    login,
    logout,
    loading,
    isAdmin: user?.role === 'admin'
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};