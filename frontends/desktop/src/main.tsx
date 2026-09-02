import './platform';
import '@semi-css';
import './global.css';
import './stores/bridgeActivity';
import { handleRenderedContentLinkClick } from './lib/rendered-content-policy';

if (document.documentElement.dataset.appearance === 'dark') {
  document.body.setAttribute('theme-mode', 'dark');
}

document.addEventListener('click', (event) => {
  const opener = (window as any).__TAURI__?.opener;
  handleRenderedContentLinkClick(
    event,
    typeof opener?.openUrl === 'function' ? (url) => opener.openUrl(url) : undefined,
  );
});

setTimeout(() => {
  document.body.classList.remove('no-transition');
}, 0);

import React from 'react';
import { Button, Collapse, Empty, Typography } from '@douyinfe/semi-ui';
import { IconRefresh } from '@douyinfe/semi-icons';
import { IllustrationFailure, IllustrationFailureDark } from '@douyinfe/semi-illustrations';
import { createRoot } from 'react-dom/client';
import { App } from './App';
import { bootLang, t } from './i18n/t';

class RootErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { error: Error | null }
> {
  state: { error: Error | null } = { error: null };

  static getDerivedStateFromError(error: Error) {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[RootErrorBoundary] React crashed:', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      const lang = bootLang();
      const tr = (key: string) => t(lang, key);
      const error = this.state.error;
      return (
        <main className="ga-root-error" role="alert">
          <Empty
            className="ga-root-error-result"
            image={<IllustrationFailure />}
            darkModeImage={<IllustrationFailureDark />}
            title={tr('crash.title')}
            description={tr('crash.description')}
          >
            <Button
              type="primary"
              theme="solid"
              icon={<IconRefresh />}
              onClick={() => window.location.reload()}
            >
              {tr('crash.reload')}
            </Button>
          </Empty>
          <Collapse className="ga-root-error-details" accordion>
            <Collapse.Panel itemKey="technical-details" header={tr('crash.details')}>
              <Typography.Paragraph type="tertiary">
                {tr('crash.detailsHint')}
              </Typography.Paragraph>
              <pre tabIndex={0}>{[error.message, error.stack].filter(Boolean).join('\n\n')}</pre>
            </Collapse.Panel>
          </Collapse>
        </main>
      );
    }
    return this.props.children;
  }
}

createRoot(document.getElementById('app')!).render(
  <RootErrorBoundary>
    <App />
  </RootErrorBoundary>,
);
