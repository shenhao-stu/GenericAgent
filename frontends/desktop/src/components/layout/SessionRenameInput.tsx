import { useEffect, useRef, useState } from 'react';
import { useI18n } from '../../i18n';

interface Props {
  initial: string;
  /** Receives the trimmed value (possibly empty); the caller decides whether that is a rename. */
  onConfirm: (title: string) => void;
  onCancel: () => void;
  className?: string;
}

/**
 * Inline title editor shared by the sidebar row and the thread header: focused and selected on mount,
 * Enter confirms, Escape cancels, leaving the field confirms. IME composition never triggers either.
 */
export function SessionRenameInput({ initial, onConfirm, onCancel, className }: Props) {
  const { t } = useI18n();
  const [value, setValue] = useState(initial);
  const inputRef = useRef<HTMLInputElement>(null);
  // Escape unmounts the field; a blur fired by that unmount must not turn the cancel into a rename.
  const settledRef = useRef(false);

  useEffect(() => {
    inputRef.current?.focus();
    inputRef.current?.select();
  }, []);

  const settle = (action: () => void) => {
    if (settledRef.current) return;
    settledRef.current = true;
    action();
  };

  return (
    <input
      ref={inputRef}
      className={className}
      value={value}
      aria-label={t('session.rename')}
      onChange={(e) => setValue(e.target.value)}
      onKeyDown={(e) => {
        if (e.nativeEvent.isComposing || e.keyCode === 229) return;
        if (e.key === 'Enter') {
          e.preventDefault();
          settle(() => onConfirm(value.trim()));
        } else if (e.key === 'Escape') {
          e.preventDefault();
          settle(onCancel);
        }
      }}
      onBlur={() => settle(() => onConfirm(value.trim()))}
      onClick={(e) => e.stopPropagation()}
    />
  );
}
