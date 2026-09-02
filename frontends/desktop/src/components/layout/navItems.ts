import type { PageId } from '../../stores/app';

/** Pages reachable from the sidebar besides chat, in display order; shared by the full sidebar and the collapsed rail. */
export const NAV_ITEMS: readonly { key: PageId; icon: string; textKey: string }[] = [
  { key: 'services', icon: 'symbol-misc', textKey: 'nav.services' },
  { key: 'collab', icon: 'robot', textKey: 'nav.collab' },
  { key: 'token', icon: 'graph', textKey: 'nav.token' },
];
