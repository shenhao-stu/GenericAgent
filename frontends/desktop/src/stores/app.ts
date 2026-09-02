import { create } from 'zustand';

export type PageId = 'chat' | 'services' | 'collab' | 'token';

const SIDEBAR_COLLAPSED_KEY = 'ga_sidebar_collapsed';

/** Sidebar collapse is a layout preference, so it survives restarts like the other boot-cached settings. */
export function readSidebarCollapsed(storage: Pick<Storage, 'getItem'> | null = safeStorage()): boolean {
  try {
    return storage?.getItem(SIDEBAR_COLLAPSED_KEY) === '1';
  } catch {
    return false;
  }
}

export function writeSidebarCollapsed(collapsed: boolean, storage: Pick<Storage, 'setItem'> | null = safeStorage()): void {
  try {
    storage?.setItem(SIDEBAR_COLLAPSED_KEY, collapsed ? '1' : '0');
  } catch { /* private mode / quota: the preference just does not persist */ }
}

function safeStorage(): Storage | null {
  try {
    return typeof localStorage === 'undefined' ? null : localStorage;
  } catch {
    return null;
  }
}

interface AppState {
  activePage: PageId;
  sidebarCollapsed: boolean;
  servicesTab: string;
  setPage: (page: PageId) => void;
  setServicesTab: (tab: string) => void;
  toggleSidebar: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  activePage: 'chat',
  sidebarCollapsed: readSidebarCollapsed(),
  // The nav item is named 运行状态 / Runtime status, so that is the tab it lands on.
  servicesTab: 'status',

  setPage: (page) => set({ activePage: page }),
  setServicesTab: (tab) => set({ servicesTab: tab }),
  toggleSidebar: () => set((s) => {
    const sidebarCollapsed = !s.sidebarCollapsed;
    writeSidebarCollapsed(sidebarCollapsed);
    return { sidebarCollapsed };
  }),
}));
