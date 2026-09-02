import { useCallback, useRef } from 'react';
import { useChatStore, type SendOptions } from '../../stores/chat';
import { profileIsMixin, useSettingsStore } from '../../stores/settings';
import { useBridgeStatus } from '../../hooks/useBridgeStatus';
import { useI18n } from '../../i18n';
import { Thread } from './Thread';
import { Composer } from './Composer';
import { WorkDirPill } from './Composer/WorkDirPill';
import { usePlaceholder } from './Composer/usePlaceholder';
import { EmptyState } from './EmptyState';
import type { SkillDef } from './Composer/skills';
import type { RichEditorHandle } from './Composer/RichEditorInput';
import './chatView.css';

export function ChatView() {
  const { t } = useI18n();
  const bridgeStatus = useBridgeStatus();
  const status = useChatStore((s) => s.status);
  const messages = useChatStore((s) => s.messages);
  const activeSessionId = useChatStore((s) => s.activeSessionId);
  const sendMessage = useChatStore((s) => s.sendMessage);
  const cancel = useChatStore((s) => s.cancel);
  const sessionModelNo = useChatStore((s) => s.sessionModelNo);
  // Aggregation backends cannot take image payloads; the session binding wins over the default.
  const imagesAsPaths = useSettingsStore((s) => profileIsMixin(s.modelProfiles, sessionModelNo ?? s.defaultModelNo));
  const { text: placeholder } = usePlaceholder();
  const composerEditorRef = useRef<RichEditorHandle>(null);

  const handleSend = useCallback(
    (text: string, opts?: SendOptions) => {
      if (text || opts) sendMessage(text, opts);
    },
    [sendMessage],
  );

  const handlePresetClick = useCallback((skill: SkillDef) => {
    composerEditorRef.current?.setSkillChip(skill.id, skill.prompt);
    composerEditorRef.current?.focus();
  }, []);

  const isEmpty = messages.length === 0 && status === 'idle';
  const showOffline = isEmpty && bridgeStatus !== 'ready';

  return (
    <div className="chat-view-root" data-empty={isEmpty || undefined}>
      {showOffline ? (
        <div className="ga-chat-offline">
          <span>{bridgeStatus === 'connecting' ? t('bridge.connecting') : t('bridge.offline')}</span>
        </div>
      ) : isEmpty ? (
        <EmptyState onPresetClick={handlePresetClick} />
      ) : (
        <Thread />
      )}
      <Composer
        sessionId={activeSessionId}
        placeholder={placeholder}
        onSend={handleSend}
        onStop={cancel}
        isGenerating={status === 'running'}
        imagesAsPaths={imagesAsPaths}
        editorRef={composerEditorRef}
        leading={<WorkDirPill />}
      />
    </div>
  );
}
