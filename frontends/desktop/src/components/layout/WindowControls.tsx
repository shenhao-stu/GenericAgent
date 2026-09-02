import { t, type Lang } from '../../i18n/t';

const CONTROLS = [
  {
    key: 'win.minimize',
    action: 'minimize',
    icon: <svg width="10" height="1" viewBox="0 0 10 1"><rect fill="currentColor" width="10" height="1" /></svg>,
  },
  {
    key: 'win.maximize',
    action: 'toggleMaximize',
    icon: (
      <svg width="10" height="10" viewBox="0 0 10 10" fill="none">
        <rect x="0.5" y="0.5" width="9" height="9" stroke="currentColor" strokeWidth="1" />
      </svg>
    ),
  },
  {
    key: 'win.close',
    action: 'close',
    icon: <svg width="10" height="10" viewBox="0 0 10 10"><path d="M1 1l8 8M9 1l-8 8" stroke="currentColor" strokeWidth="1.2" /></svg>,
  },
] as const;

/** Store-free so the loading page can share it; the caller supplies the language. */
export function WindowControls({ lang }: { lang: Lang }) {
  const win = (window as any).__TAURI__?.window?.getCurrentWindow?.();
  if (!win) return null;

  return (
    <div className="ga-win-controls" data-no-drag>
      {CONTROLS.map(({ key, action, icon }) => (
        <button
          key={key}
          type="button"
          className={`ga-win-btn${action === 'close' ? ' ga-win-btn--close' : ''}`}
          onClick={() => win[action]()}
          aria-label={t(lang, key)}
          title={t(lang, key)}
        >
          {icon}
        </button>
      ))}
    </div>
  );
}
