import { presetColor, presetIcon, presetLabel, type ProviderPreset } from '../../data/model-presets';
import { useI18n } from '../../i18n';

interface Props {
  preset: ProviderPreset;
  onClick: () => void;
}

export function ProviderCard({ preset, onClick }: Props) {
  const { t, lang } = useI18n();
  const Icon = presetIcon(preset);

  return (
    <button
      type="button"
      className="ga-provider-card"
      onClick={onClick}
      style={{ '--provider-color': presetColor(preset) } as React.CSSProperties}
    >
      <span className="ga-provider-card-icon"><Icon size={20} /></span>
      <span className="ga-provider-card-body">
        <span className="ga-provider-card-label">{presetLabel(preset, lang)}</span>
        <span className="ga-provider-card-desc">{t(preset.descKey)}</span>
      </span>
      <span className="ga-provider-card-caret">›</span>
    </button>
  );
}
