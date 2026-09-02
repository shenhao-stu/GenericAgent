import { useEffect, useState, useCallback } from 'react';
import { Modal } from '@douyinfe/semi-ui';
import { Codicon } from '../../lib/icons';
import { SETTINGS_SECTIONS, settingsSectionOf, useSettingsStore, type SettingsSection } from '../../stores/settings';
import { useI18n } from '../../i18n';
import './settings.css';
import { AppearanceSection } from './AppearanceSection';
import { LanguageSection } from './LanguageSection';
import { ModelSection } from './ModelSection';
import { DataSection } from './DataSection';
import { ConnectionModeSection } from './ConnectionModeSection';
import { HelpFeedbackSection } from './HelpFeedbackSectionView';
import { AddModelView } from './AddModelView';

/** One entry per section: the nav label reuses the section's own heading so a concept has exactly one name. */
const SECTION_META: Record<SettingsSection, { labelKey: string; icon: string }> = {
  general: { labelKey: 'set.general', icon: 'settings' },
  models: { labelKey: 'set.model', icon: 'hubot' },
  data: { labelKey: 'data.title', icon: 'database' },
  connection: { labelKey: 'connection.title', icon: 'plug' },
  help: { labelKey: 'helpFeedback.title', icon: 'question' },
};

export function SettingsModal() {
  const { visible, view, close, setView, loadFromBridge } = useSettingsStore();
  const { t } = useI18n();
  const [editingId, setEditingId] = useState<number | null>(null);
  const section = settingsSectionOf(view);

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
    setView('models');
    setEditingId(null);
  }, [setView]);

  const title = view !== 'addModel'
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
      <div className="ga-settings-layout">
        <nav className="ga-settings-nav" aria-label={t('modal.settings')}>
          {SETTINGS_SECTIONS.map((key) => (
            <button
              key={key}
              type="button"
              className={`ga-settings-nav-btn${key === section ? ' active' : ''}`}
              aria-current={key === section ? 'page' : undefined}
              onClick={() => { setEditingId(null); setView(key); }}
            >
              <Codicon name={SECTION_META[key].icon} size="1rem" />
              <span>{t(SECTION_META[key].labelKey)}</span>
            </button>
          ))}
        </nav>
        <div className="ga-settings-pane" data-section={section}>
          {view === 'addModel' ? <AddModelView editingId={editingId} onDone={handleModelDone} /> : (
            <>
              {section === 'general' && <><AppearanceSection /><LanguageSection /></>}
              {section === 'models' && <ModelSection onAdd={handleAddModel} onEdit={handleEditModel} />}
              {section === 'data' && <DataSection />}
              {section === 'connection' && <ConnectionModeSection />}
              {section === 'help' && <HelpFeedbackSection />}
            </>
          )}
        </div>
      </div>
    </Modal>
  );
}
