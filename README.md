# Physical AI & Humanoid Robotics Textbook

A comprehensive textbook for teaching Physical AI & Humanoid Robotics course, focusing on AI Systems in the Physical World and Embodied Intelligence.

## Features

- **Docusaurus v3+**: Modern documentation website framework
- **Admin Dashboard**: Production-ready admin interface for content management
- **RAG Chatbot**: AI-powered chatbot with retrieval-augmented generation
- **Authentication**: Better Auth integration for secure access
- **Internationalization**: English and Urdu language support
- **Vercel Deploy Ready**: Optimized for Vercel deployment

## Course Overview

The future of AI extends beyond digital spaces into the physical world. This capstone course introduces Physical AI—AI systems that function in reality and comprehend physical laws. Students learn to design, simulate, and deploy humanoid robots capable of natural human interactions using ROS 2, Gazebo, and NVIDIA Isaac.

### Module 1: The Robotic Nervous System (ROS 2)
- ROS 2 Nodes, Topics, and Services
- Bridging Python Agents to ROS controllers using rclpy
- Understanding URDF (Unified Robot Description Format) for humanoids

### Module 2: The Digital Twin (Gazebo & Unity)
- Simulating physics, gravity, and collisions in Gazebo
- High-fidelity simulation environments
- Unity integration for advanced visualization

## Prerequisites

- Node.js (v18 or higher)
- npm or yarn

## Local Development

1. Install dependencies:
   ```bash
   npm install
   ```

2. Create environment variables file:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. Start the development server:
   ```bash
   npm start
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Environment Variables

Create a `.env.local` file in the root directory with the following variables:

```env
NEXT_PUBLIC_API_URL=
NEXT_PUBLIC_AUTH_URL=
NEXT_PUBLIC_CHAT_API_URL=
NEXT_PUBLIC_SITE_URL=
AUTH_SECRET=your-super-secret-key-here
DATABASE_URL=file:./sqlite.db
```

## Authentication

The application uses Better Auth for authentication:

- **Login**: Navigate to `/login` or use the login button in the navbar
- **Register**: Navigate to `/register` to create a new account
- **Protected Routes**: Admin routes (`/admin/*`) are protected and require authentication
- **Admin Access**: Only users with the `admin` role can access the admin dashboard

### Admin Access

To access the admin dashboard:
1. Register an account or login
2. Navigate to `/admin`
3. Only users with admin role can access the dashboard

## Internationalization

This site supports English and Urdu languages with user-controlled translation:
- English: Default language at `/`
- Urdu: User-controlled translation via translate buttons

Users can toggle between English and Urdu content using the translation buttons available on each page. The content personalization feature allows logged-in users to customize their learning experience based on their language preference.

## RAG Chatbot

The AI assistant is available as a floating widget on all pages:
- Click the chat icon in the bottom-right corner to open the chat
- Ask questions about Physical AI and Humanoid Robotics
- The chatbot uses a mock API in this frontend-only implementation
- In a production environment, connect to your actual RAG backend via the `NEXT_PUBLIC_CHAT_API_URL` environment variable

## Admin Dashboard Features

The admin dashboard includes:

1. **Dashboard Overview**: Key metrics and recent activity
2. **User Management**: View, edit, and manage user accounts
3. **Content Control**: Manage documentation and blog posts
4. **Analytics**: View site performance and chatbot usage metrics

## Vercel Deployment

This project is configured for Vercel deployment:

1. Push your code to a GitHub repository
2. Connect your repository to Vercel
3. Vercel will automatically detect the Docusaurus configuration and build settings
4. Add your environment variables in the Vercel dashboard under Settings > Environment Variables
5. Your site will be deployed automatically on pushes to the main branch

### Vercel Configuration

The `vercel.json` file contains the following configuration:
- Framework: Docusaurus
- Build command: `npm run build`
- Output directory: `build`

## Project Structure

```
/
├── README.md
├── package.json
├── docusaurus.config.js
├── sidebars.js
├── vercel.json
├── .env.example
├── src/
│   ├── components/
│   ├── pages/
│   └── css/
├── docs/
├── i18n/
│   ├── en/
│   └── ur/
└── static/
```

## Available Scripts

- `npm start`: Start the development server
- `npm build`: Build the site for production
- `npm serve`: Serve the built site locally
- `npm deploy`: Deploy to GitHub Pages (if configured)
- `npm write-translations`: Extract and update translation files for i18n

## License

This project is licensed under the MIT License.