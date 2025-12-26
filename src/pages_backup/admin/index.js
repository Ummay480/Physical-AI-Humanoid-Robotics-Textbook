import React from 'react';
import Layout from '@theme/Layout';
import { useAuth } from '../../components/AuthContext';

function AdminDashboard() {
  const { user, logout } = useAuth();

  if (!user) {
    // Redirect to login if not authenticated
    window.location.href = '/login';
    return null;
  }

  if (user.role !== 'admin') {
    return (
      <Layout title="Access Denied" description="You don't have permission to access this page">
        <div className="container margin-vert--lg">
          <div className="row">
            <div className="col col--6 col--offset-3">
              <div className="admin-card">
                <h1>Access Denied</h1>
                <p>You need admin privileges to access this page.</p>
                <a href="/">Go back to home</a>
              </div>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  const handleLogout = () => {
    logout();
    window.location.href = '/';
  };

  return (
    <Layout title="Admin Dashboard" description="Admin dashboard for managing content and users">
      <div className="admin-container">
        <div className="admin-header">
          <h1>Admin Dashboard</h1>
          <p>Welcome, {user.name || user.email}!</p>
        </div>

        <div className="row">
          <div className="col col--3">
            <div className="admin-card">
              <h3>Navigation</h3>
              <ul className="menu">
                <li><a href="/admin">Dashboard</a></li>
                <li><a href="/admin/users">User Management</a></li>
                <li><a href="/admin/content">Content Control</a></li>
                <li><a href="/admin/analytics">Analytics</a></li>
              </ul>
              <button
                className="button button--secondary margin-top--md"
                onClick={handleLogout}
              >
                Logout
              </button>
            </div>
          </div>

          <div className="col col--9">
            <div className="admin-card">
              <h2>Dashboard Overview</h2>
              <p>Manage your content, users, and analytics from this dashboard.</p>

              <div className="row margin-vert--lg">
                <div className="col col--4">
                  <div className="admin-card">
                    <h3>Users</h3>
                    <p>124</p>
                    <a href="/admin/users">Manage Users</a>
                  </div>
                </div>
                <div className="col col--4">
                  <div className="admin-card">
                    <h3>Documents</h3>
                    <p>56</p>
                    <a href="/admin/content">Manage Content</a>
                  </div>
                </div>
                <div className="col col--4">
                  <div className="admin-card">
                    <h3>Chatbot Sessions</h3>
                    <p>1,243</p>
                    <a href="/admin/analytics">View Analytics</a>
                  </div>
                </div>
              </div>
            </div>

            <div className="admin-card">
              <h3>Recent Activity</h3>
              <table className="table">
                <thead>
                  <tr>
                    <th>User</th>
                    <th>Action</th>
                    <th>Time</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>John Doe</td>
                    <td>Updated documentation</td>
                    <td>2 minutes ago</td>
                  </tr>
                  <tr>
                    <td>Jane Smith</td>
                    <td>Published new article</td>
                    <td>15 minutes ago</td>
                  </tr>
                  <tr>
                    <td>Bob Johnson</td>
                    <td>Commented on article</td>
                    <td>1 hour ago</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
}

export default AdminDashboard;