import { useMemo } from 'react';
import { useSettingsStore } from '../stores/settings';
import { t } from './t';

export { t } from './t';
export type { Lang } from './t';

export function useI18n() {
  const lang = useSettingsStore((s) => s.lang);
  return useMemo(
    () => ({
      lang,
      t: (key: string, params?: Record<string, string | number>) => t(lang, key, params),
    }),
    [lang],
  );
}
