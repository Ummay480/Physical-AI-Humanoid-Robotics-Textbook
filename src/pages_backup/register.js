import React, { useState } from 'react';
import Layout from '@theme/Layout';
import { useAuth } from '../components/AuthContext';

function RegisterPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [background, setBackground] = useState('');
  const [error, setError] = useState('');
  const { login, user } = useAuth(); // Using login to set user after registration

  if (user) {
    window.location.href = '/'; // Redirect if already logged in
    return null;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();

    // In a real app, you would send this to your backend
    // For demo purposes, we'll just log the background info
    console.log('User background info:', { background });

    // Simulate registration by logging in with the provided credentials
    try {
      await login(email, password);
      window.location.href = '/'; // Redirect to home after registration
    } catch (err) {
      setError('Registration failed. Please try again.');
    }
  };

  return (
    <Layout title="Register" description="Create a new account">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--6 col--offset-3">
            <div className="admin-card">
              <h1>Register</h1>
              {error && (
                <div className="alert alert--danger">
                  {error}
                </div>
              )}
              <form onSubmit={handleSubmit}>
                <div className="margin-bottom--lg">
                  <label htmlFor="name">Full Name</label>
                  <input
                    type="text"
                    id="name"
                    className="form-control"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                </div>
                <div className="margin-bottom--lg">
                  <label htmlFor="email">Email</label>
                  <input
                    type="email"
                    id="email"
                    className="form-control"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                  />
                </div>
                <div className="margin-bottom--lg">
                  <label htmlFor="password">Password</label>
                  <input
                    type="password"
                    id="password"
                    className="form-control"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                  />
                </div>
                <div className="margin-bottom--lg">
                  <label htmlFor="background">Software/Hardware Background</label>
                  <textarea
                    id="background"
                    className="form-control"
                    value={background}
                    onChange={(e) => setBackground(e.target.value)}
                    placeholder="Please describe your background in software development, robotics, or hardware. This will help us personalize your learning experience."
                    required
                  />
                </div>
                <button type="submit" className="button button--primary button--block">
                  Register
                </button>
              </form>
              <div className="margin-top--md">
                <p>Already have an account? <a href="/login">Login here</a></p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default RegisterPage;