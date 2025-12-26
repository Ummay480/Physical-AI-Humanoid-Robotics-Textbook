import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import ChatbotWidget from '../components/ChatbotWidget';
import ContentPersonalization from '../components/ContentPersonalization';
import { AuthProvider } from '../components/AuthContext';
import Translate, {translate} from '@docusaurus/Translate';

import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          <Translate id="homepage.title">Physical AI & Humanoid Robotics</Translate>
        </Heading>
        <p className="hero__subtitle">
          <Translate id="homepage.tagline">Bridging the gap between digital brain and physical body</Translate>
        </p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            <Translate id="homepage.buttonText">Read the Textbook - 5min ⏱️</Translate>
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
        title={translate({id: 'homepage.meta.title', message: 'Welcome to Physical AI & Humanoid Robotics'})}
        description={translate({id: 'homepage.meta.description', message: 'Physical AI & Humanoid Robotics textbook'})}>
        <HomepageHeader />
        <main>
          <ContentPersonalization contentKey="homepage-features">
            <section className={styles.features}>
              <div className="container">
                <div className="row">
                  <div className="col col--4">
                    <div className="text--center">
                      <h3><Translate id="homepage.feature1.title">Physical AI</Translate></h3>
                    </div>
                    <div className="text--center padding-horiz--md">
                      <p><Translate id="homepage.feature1.description">Learn about AI systems that function in the real physical world and comprehend physical laws.</Translate></p>
                    </div>
                  </div>
                  <div className="col col--4">
                    <div className="text--center">
                      <h3><Translate id="homepage.feature2.title">Embodied Intelligence</Translate></h3>
                    </div>
                    <div className="text--center padding-horiz--md">
                      <p><Translate id="homepage.feature2.description">Discover how intelligence emerges from the interaction between an agent and its physical environment.</Translate></p>
                    </div>
                  </div>
                  <div className="col col--4">
                    <div className="text--center">
                      <h3><Translate id="homepage.feature3.title">Humanoid Robotics</Translate></h3>
                    </div>
                    <div className="text--center padding-horiz--md">
                      <p><Translate id="homepage.feature3.description">Explore the design, control, and applications of robots with human-like form and behavior.</Translate></p>
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