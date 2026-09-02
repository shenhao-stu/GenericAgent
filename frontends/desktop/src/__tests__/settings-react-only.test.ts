// @vitest-environment happy-dom
import { beforeEach, describe, expect, it, vi } from 'vitest';

const mockSaveConfig = vi.fn().mockResolvedValue(undefined);

vi.mock('../services/bridge', () => ({
  getConfig: vi.fn(),
  getModelProfiles: vi.fn(),
  saveConfig: (...args: unknown[]) => mockSaveConfig(...args),
}));

import { hasUsableModel, settingsSectionOf, useSettingsStore } from '../stores/settings';

const mixin = { id: 0, name: '默认组', model: '', apibase: '', protocol: 'oai' as const, stream: true, kind: 'mixin' as const, members: [] };
const native = { id: 1, name: 'Model A', model: 'model-a', apibase: 'http://localhost', protocol: 'oai' as const, stream: true };

describe('settings dialog view + first-run predicate', () => {
  it('open() targets a view and close() resets it', () => {
    const store = useSettingsStore.getState();
    store.open('addModel');
    expect(useSettingsStore.getState()).toMatchObject({ visible: true, view: 'addModel' });
    store.close();
    expect(useSettingsStore.getState()).toMatchObject({ visible: false, view: 'general' });
    store.open();
    expect(useSettingsStore.getState().view).toBe('general');
    store.open('models');
    expect(useSettingsStore.getState().view).toBe('models');
  });

  it('the add-model form highlights the models section in the nav', () => {
    expect(settingsSectionOf('addModel')).toBe('models');
    expect(settingsSectionOf('data')).toBe('data');
  });

  it('only a concrete provider profile counts as a usable model', () => {
    expect(hasUsableModel([])).toBe(false);
    expect(hasUsableModel([mixin])).toBe(false);
    expect(hasUsableModel([mixin, native])).toBe(true);
  });

  it('profilesLoaded flips only after a real profile load', () => {
    useSettingsStore.setState({ profilesLoaded: false, modelProfiles: [] });
    expect(useSettingsStore.getState().profilesLoaded).toBe(false);
    useSettingsStore.getState().setModelProfiles([mixin]);
    expect(useSettingsStore.getState().profilesLoaded).toBe(true);
  });
});

describe('React-only settings persistence', () => {
  beforeEach(() => {
    localStorage.clear();
    mockSaveConfig.mockClear();
    delete (window as Window & { gaLegacy?: unknown }).gaLegacy;
    useSettingsStore.setState({
      appearance: 'light',
      chatFontSize: 14,
      lang: 'zh',
      defaultModelNo: 0,
      modelProfiles: [
        {
          id: 0,
          name: 'Model A',
          model: 'model-a',
          apibase: 'http://localhost',
          protocol: 'oai',
          stream: true,
        },
        {
          id: 1,
          name: 'Model B',
          model: 'model-b',
          apibase: 'http://localhost',
          protocol: 'oai',
          stream: true,
        },
      ],
    });
  });

  it('applies and persists settings without a v1 gaLegacy global', async () => {
    const store = useSettingsStore.getState();

    expect(() => {
      store.setAppearance('dark');
      store.setLang('en');
      store.setDefaultModel(1);
    }).not.toThrow();

    await vi.waitFor(() => expect(mockSaveConfig).toHaveBeenCalledTimes(3));
    expect(document.documentElement.dataset.appearance).toBe('dark');
    expect(document.documentElement.lang).toBe('en');
    expect(document.body.getAttribute('theme-mode')).toBe('dark');
    expect(localStorage.getItem('ga_appearance')).toBe('dark');
    expect(localStorage.getItem('ga_lang')).toBe('en');
    expect(localStorage.getItem('ga_llm_no')).toBe('1');
    expect(useSettingsStore.getState().defaultModelNo).toBe(1);
  });
});
