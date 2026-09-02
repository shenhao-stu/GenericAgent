import { useEffect, useState, useCallback } from 'react';
import { Modal } from '@douyinfe/semi-ui';
import { useSettingsStore } from '../../stores/settings';
import { useI18n } from '../../i18n';
import './settings.css';
import { AppearanceSection } from './AppearanceSection';
import { LanguageSection } from './LanguageSection';
import { ModelSection } from './ModelSection';
import { DataSection } from './DataSection';
import { ConnectionModeSection } from './ConnectionModeSection';
import { HelpFeedbackSection } from './HelpFeedbackSectionView';
import { AddModelView } from './AddModelView';

export function SettingsModal() {
  const { visible, view, close, setView, loadFromBridge } = useSettingsStore();
  const { t } = useI18n();

  const [editingId, setEditingId] = useState<number | null>(null);

  useEffect(() => {
    if (visible) loadFromBridge();
    else setEditingId(null);
  }, [visible, loadFromBridge]);

  const handleAddModel = useCallback(() => {
    setEditingId(null);
    setView('addModel');
  }, [setView]);

  const handleEditModel = useCallback((id: number) => {
    setEditingId(id);
    setView('addModel');
  }, [setView]);

  const handleModelDone = useCallback(() => {
    setView('main');
    setEditingId(null);
  }, [setView]);

  const title = view === 'main'
    ? t('modal.settings')
    : (editingId != null ? t('modal.editModel') : t('modal.addModel'));

  return (
    <Modal
      visible={visible}
      onCancel={close}
      title={title}
      footer={null}
      width={870}
      centered
      closeOnEsc
      className="ga-settings-dialog"
    >
      {view === 'main' ? (
        <>
          <AppearanceSection />
          <LanguageSection />
          <ModelSection onAdd={handleAddModel} onEdit={handleEditModel} />
          <DataSection />
          <ConnectionModeSection />
          <HelpFeedbackSection />
        </>
      ) : (
        <AddModelView editingId={editingId} onDone={handleModelDone} />
      )}
    </Modal>
  );
}
