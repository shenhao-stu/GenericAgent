import { useCallback, useEffect } from 'react';
import { useConductorStore } from '../../stores/conductor';
import { profileIsMixin, useSettingsStore } from '../../stores/settings';
import { useI18n } from '../../i18n';
import { Composer } from '../chat/Composer';
import { ModelSelector } from '../chat/Composer/ModelSelector';
import type { SendOptions } from '../../stores/chat';

export function CollabComposer() {
  const { t } = useI18n();
  const sendMessage = useConductorStore((s) => s.sendMessage);
  const conductorTyping = useConductorStore((s) => s.conductorTyping);
  const connectionStatus = useConductorStore((s) => s.connectionStatus);
  const modelConfig = useConductorStore((s) => s.modelConfig);
  const runtimeModel = useConductorStore((s) => s.runtimeModel);
  const loadModel = useConductorStore((s) => s.loadModel);
  const selectModel = useConductorStore((s) => s.selectModel);
  const defaultModelNo = useSettingsStore((s) => s.defaultModelNo);
  const selectedNo = modelConfig?.effective ?? modelConfig?.configured ?? defaultModelNo;
  const imagesAsPaths = useSettingsStore((s) => profileIsMixin(s.modelProfiles, selectedNo));

  useEffect(() => { loadModel(); }, [loadModel]);

  const handleSend = useCallback((text: string, opts?: SendOptions) => {
    const files = opts?.files?.map((f) => ({ name: f.name, path: f.path }));
    const images = opts?.images?.map((f) => ({ name: f.name, path: f.path, base64: f.base64 }));
    sendMessage(text, files, images);
  }, [sendMessage]);

  const disabled = connectionStatus !== 'ready';

  return (
    <div className="collab-composer-wrap" data-slot="collab-composer" data-disabled={disabled || undefined}>
      <Composer
        sessionId="__collab_composer__"
        placeholder={t('collab.placeholder')}
        onSend={handleSend}
        onStop={() => {}}
        isGenerating={conductorTyping}
        canStop={false}
        imagesAsPaths={imagesAsPaths}
        hideStatusStack
        modelControl={(
          <ModelSelector
            selectedNo={selectedNo}
            runningNo={runtimeModel?.running ? runtimeModel.effective : null}
            isRunning={!!runtimeModel?.running}
            onSelect={selectModel}
          />
        )}
      />
    </div>
  );
}
