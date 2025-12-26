import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import ChatbotWidget from '../../../src/components/ChatbotWidget';
import ContentPersonalization from '../../../src/components/ContentPersonalization';
import { AuthProvider } from '../../../src/components/AuthContext';

import Heading from '@theme/Heading';
import styles from '../../../src/pages/index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <Heading as="h1" className="hero__title">
          فزیکل AI اور ہیومنائیڈ روبوٹکس
        </Heading>
        <p className="hero__subtitle">ڈیجیٹل دماغ اور جسمانی جسم کے درمیان خلا کو ختم کرنا</p>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            کتاب پڑھیں - 5 منٹ ⏱️
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
        title={`خوش آمدید`}
        description="فزیکل AI اور ہیومنائیڈ روبوٹکس کی کتاب">
        <HomepageHeader />
        <main>
          <ContentPersonalization contentKey="homepage-features">
            <section className={styles.features}>
              <div className="container">
                <div className="row">
                  <div className="col col--4">
                    <div className="text--center">
                      <h3>فزیکل AI</h3>
                    </div>
                    <div className="text--center padding-horiz--md">
                      <p>AI سسٹمز کے بارے میں سیکھیں جو حقیقی جسمانی دنیا میں کام کرتے ہیں اور فزیکل قوانین کو سمجھتے ہیں۔</p>
                    </div>
                  </div>
                  <div className="col col--4">
                    <div className="text--center">
                      <h3>مجسم ذہانت</h3>
                    </div>
                    <div className="text--center padding-horiz--md">
                      <p>دریافت کریں کہ کس طرح ایک ایجنٹ اور اس کے جسمانی ماحول کے درمیان تعامل سے ذہانت پیدا ہوتی ہے۔</p>
                    </div>
                  </div>
                  <div className="col col--4">
                    <div className="text--center">
                      <h3>ہیومنائیڈ روبوٹکس</h3>
                    </div>
                    <div className="text--center padding-horiz--md">
                      <p>انسان نما شکل اور رویے کے ساتھ روبوٹس کے ڈیزائن، کنٹرول اور ایپلی کیشنز کو دریافت کریں۔</p>
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
