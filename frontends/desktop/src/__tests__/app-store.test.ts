// @vitest-environment happy-dom
import { beforeEach, describe, expect, it } from 'vitest';
import { readSidebarCollapsed, useAppStore, writeSidebarCollapsed } from '../stores/app';
import { NAV_ITEMS } from '../components/layout/navItems';
import { t } from '../i18n/t';

describe('app store', () => {
  beforeEach(() => {
    localStorage.clear();
    useAppStore.setState({ sidebarCollapsed: false, activePage: 'chat' });
  });

  it('remembers the sidebar collapse across launches through the boot cache', () => {
    expect(readSidebarCollapsed()).toBe(false);
    useAppStore.getState().toggleSidebar();
    expect(useAppStore.getState().sidebarCollapsed).toBe(true);
    expect(readSidebarCollapsed()).toBe(true);
    useAppStore.getState().toggleSidebar();
    expect(readSidebarCollapsed()).toBe(false);
  });

  it('treats an unavailable or throwing storage as "not collapsed"', () => {
    expect(readSidebarCollapsed(null)).toBe(false);
    const broken = { getItem: () => { throw new Error('denied'); }, setItem: () => { throw new Error('denied'); } };
    expect(readSidebarCollapsed(broken)).toBe(false);
    expect(() => writeSidebarCollapsed(true, broken)).not.toThrow();
  });

  it('lands runtime status on the status tab and keeps every rail destination localized', () => {
    expect(useAppStore.getState().servicesTab).toBe('status');
    for (const item of NAV_ITEMS) {
      expect(t('zh', item.textKey)).not.toBe(item.textKey);
      expect(t('en', item.textKey)).not.toBe(item.textKey);
    }
  });
});
