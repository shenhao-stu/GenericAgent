import { create } from 'zustand';

export type PageId = 'chat' | 'services' | 'collab' | 'token';

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
  sidebarCollapsed: false,
  // The nav item is named 运行状态 / Runtime status, so that is the tab it lands on.
  servicesTab: 'status',

  setPage: (page) => set({ activePage: page }),
  setServicesTab: (tab) => set({ servicesTab: tab }),
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
}));
