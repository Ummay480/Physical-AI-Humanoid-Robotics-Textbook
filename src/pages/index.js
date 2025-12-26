import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import ChatbotWidget from '../components/ChatbotWidget';
import ContentPersonalization from '../components/ContentPersonalization';
import { AuthProvider } from '../components/AuthContext';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          {siteConfig.title}
        </Heading>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Read the Textbook - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <AuthProvider>
      <Layout
        title={`Welcome to ${siteConfig.title}`}
        description="Physical AI & Humanoid Robotics textbook">
        <HomepageHeader />
        <main>
          <ContentPersonalization contentKey="homepage-features">
            <section className={styles.features}>
              <div className="container">
                <div className="row">
                  <div className="col col--4">
                    <div className="text--center">
                      <h3>Physical AI</h3>
                    </div>
                    <div className="text--center padding-horiz--md">
                      <p>Learn about AI systems that function in the real physical world and comprehend physical laws.</p>
                    </div>
                  </div>
                  <div className="col col--4">
                    <div className="text--center">
                      <h3>Embodied Intelligence</h3>
                    </div>
                    <div className="text--center padding-horiz--md">
                      <p>Discover how intelligence emerges from the interaction between an agent and its physical environment.</p>
                    </div>
                  </div>
                  <div className="col col--4">
                    <div className="text--center">
                      <h3>Humanoid Robotics</h3>
                    </div>
                    <div className="text--center padding-horiz--md">
                      <p>Explore the design, control, and applications of robots with human-like form and behavior.</p>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </ContentPersonalization>
        </main>
        <ChatbotWidget />
      </Layout>
    </AuthProvider>
  );
}