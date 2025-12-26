import React, { useState } from 'react';
import Layout from '@theme/Layout';
import { useAuth } from '../components/AuthContext';

function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const { login, user } = useAuth();

  if (user) {
    window.location.href = '/'; // Redirect if already logged in
    return null;
  }

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      await login(email, password);
      window.location.href = '/'; // Redirect to home after login
    } catch (err) {
      setError('Invalid email or password');
    }
  };

  return (
    <Layout title="Login" description="Login to your account">
      <div className="container margin-vert--lg">
        <div className="row">
          <div className="col col--6 col--offset-3">
            <div className="admin-card">
              <h1>Login</h1>
              {error && (
                <div className="alert alert--danger">
                  {error}
                </div>
              )}
              <form onSubmit={handleSubmit}>
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
                <button type="submit" className="button button--primary button--block">
                  Login
                </button>
              </form>
              <div className="margin-top--md">
                <p>Don't have an account? <a href="/register">Register here</a></p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default LoginPage;