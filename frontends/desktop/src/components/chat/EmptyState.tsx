import { IconChecklistStroked, IconFlagStroked, IconLightningStroked, IconShareStroked, IconShieldStroked, IconSearchStroked, IconPlus } from '@douyinfe/semi-icons';
import { useI18n } from '../../i18n';
import { hasUsableModel, useSettingsStore } from '../../stores/settings';
import { BUILTIN_SKILLS, skillDescription, skillTitle, type SkillDef } from './Composer/skills';
import wordmarkLight from '../../assets/generic-agent-black.svg';
import wordmarkDark from '../../assets/generic-agent-white.svg';
import './emptyState.css';

const SKILL_ICONS: Record<string, React.ReactNode> = {
  plan: <IconChecklistStroked size="small" />,
  goal: <IconFlagStroked size="small" />,
  autonomous: <IconLightningStroked size="small" />,
  hive: <IconShareStroked size="small" />,
  review: <IconShieldStroked size="small" />,
  findwork: <IconSearchStroked size="small" />,
};

interface Props {
  onPresetClick: (skill: SkillDef) => void;
}

export function EmptyState({ onPresetClick }: Props) {
  const { t, lang } = useI18n();
  const needsModel = useSettingsStore((s) => s.profilesLoaded && !hasUsableModel(s.modelProfiles));
  const openSettings = useSettingsStore((s) => s.open);

  return (
    <div data-slot="empty-state-root">
      <div data-slot="empty-state-wordmark">
        <img src={wordmarkLight} alt="GenericAgent" className="empty-state-wordmark-light" />
        <img src={wordmarkDark} alt="" className="empty-state-wordmark-dark" aria-hidden="true" />
      </div>
      {needsModel && (
        <div data-slot="empty-state-onboarding" role="status">
          <div data-slot="empty-state-onboarding-text">
            <strong>{t('onboarding.noModelTitle')}</strong>
            <span>{t('onboarding.noModelBody')}</span>
          </div>
          <button type="button" className="empty-state-onboarding-btn" onClick={() => openSettings('addModel')}>
            <IconPlus size="small" />
            <span>{t('onboarding.addModel')}</span>
          </button>
        </div>
      )}
      <div data-slot="empty-state-presets">
        {BUILTIN_SKILLS.map((skill) => (
          <button
            key={skill.id}
            type="button"
            className="empty-state-preset-btn"
            onClick={() => onPresetClick(skill)}
            title={skillDescription(skill, t, lang)}
          >
            <span className="empty-state-preset-icon">{SKILL_ICONS[skill.id]}</span>
            <span>{skillTitle(skill, t)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
