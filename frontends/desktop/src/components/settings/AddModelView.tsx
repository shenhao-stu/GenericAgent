import { useState, useEffect, useCallback } from 'react';
import { Input, InputNumber, RadioGroup, Radio, Toast, Button } from '@douyinfe/semi-ui';
import { useSettingsStore } from '../../stores/settings';
import { useI18n } from '../../i18n';
import * as bridge from '../../services/bridge';
import type { ModelProfile } from '../../services/bridge';
import { PROVIDER_PRESETS, type ProviderPreset } from '../../data/model-presets';
import { ProviderCard } from './ProviderCard';

interface Props {
  editingId: number | null;
  onDone: () => void;
}

type Route = 'provider' | 'custom';
type Step = 'choose' | 'form';

interface FormState {
  model: string;
  apikey: string;
  apibase: string;
  name: string;
  protocol: 'oai' | 'claude';
  stream: boolean;
  max_retries: number;
  connect_timeout: number;
  read_timeout: number;
}

const DEFAULTS: FormState = {
  model: '',
  apikey: '',
  apibase: '',
  name: '',
  protocol: 'oai',
  stream: true,
  max_retries: 5,
  connect_timeout: 15,
  read_timeout: 300,
};

export function AddModelView({ editingId, onDone }: Props) {
  const setModelProfiles = useSettingsStore((s) => s.setModelProfiles);
  const { t } = useI18n();

  const [step, setStep] = useState<Step>('choose');
  const [route, setRoute] = useState<Route | null>(null);
  const [preset, setPreset] = useState<ProviderPreset | null>(null);
  const [form, setForm] = useState<FormState>(DEFAULTS);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [loading, setLoading] = useState(false);
  const [probe, setProbe] = useState<{ state: 'idle' } | { state: 'running' } | { state: 'done'; result: bridge.ModelProbeResult }>({ state: 'idle' });

  const isEdit = editingId != null;
  const requiredFilled = !!form.model.trim() && !!form.apibase.trim() && (isEdit || !!form.apikey.trim());
  const updateForm = (patch: Partial<FormState>) => {
    setForm((f) => ({ ...f, ...patch }));
    setProbe({ state: 'idle' });
  };

  useEffect(() => {
    if (isEdit) {
      setStep('form');
      setRoute('custom');
      setPreset(null);
      setLoading(true);
      bridge.getModelProfileDetail(editingId!).then((profile) => {
        if (profile) {
          setForm({
            model: profile.model || '',
            apikey: profile.apikey || '',
            apibase: profile.apibase || '',
            name: profile.name || '',
            protocol: profile.protocol || 'oai',
            stream: profile.stream !== false,
            max_retries: profile.max_retries ?? 5,
            connect_timeout: profile.connect_timeout ?? 15,
            read_timeout: profile.read_timeout ?? 300,
          });
        }
        setLoading(false);
      });
    } else {
      setForm(DEFAULTS);
      setStep('choose');
      setRoute(null);
      setPreset(null);
      setShowAdvanced(false);
      setLoading(false);
    }
  }, [editingId, isEdit]);

  const handleSelectProvider = useCallback((p: ProviderPreset) => {
    setPreset(p);
    setRoute('provider');
    setForm({
      ...DEFAULTS,
      model: p.defaultModel,
      apibase: p.apibase,
      name: p.defaultName,
      protocol: p.protocol,
    });
    setStep('form');
  }, []);

  const handleSelectCustom = useCallback(() => {
    setRoute('custom');
    setPreset(null);
    setForm(DEFAULTS);
    setStep('form');
  }, []);

  const handleBack = useCallback(() => {
    setStep('choose');
    setRoute(null);
    setPreset(null);
  }, []);

  const handleProbe = async () => {
    if (!requiredFilled) {
      Toast.warning({ content: t('err.modelRequired') });
      return;
    }
    setProbe({ state: 'running' });
    try {
      const result = await bridge.probeModelProfile({
        protocol: form.protocol,
        apibase: form.apibase.trim(),
        model: form.model.trim(),
        apikey: form.apikey.trim() || undefined,
      });
      setProbe({ state: 'done', result });
    } catch (error) {
      setProbe({ state: 'done', result: { ok: false, error: error instanceof Error ? error.message : String(error) } });
    }
  };

  const handleSubmit = async () => {
    if (!requiredFilled) {
      Toast.warning({ content: t('err.modelRequired') });
      return;
    }

    try {
      const data: Partial<ModelProfile> = {
        model: form.model.trim(),
        apibase: form.apibase.trim(),
        name: form.name.trim() || undefined,
        protocol: form.protocol,
        stream: form.stream,
        max_retries: form.max_retries,
        connect_timeout: form.connect_timeout,
        read_timeout: form.read_timeout,
      };
      if (form.apikey.trim()) data.apikey = form.apikey.trim();

      let profiles: ModelProfile[];
      if (isEdit) {
        profiles = await bridge.editModelProfile(editingId!, data);
      } else {
        profiles = await bridge.addModelProfile(data);
      }
      setModelProfiles(profiles);
      Toast.success({ content: t('sys.configSaved') });
      onDone();
    } catch (error) {
      Toast.error({ content: `${t('err.modelSave')}: ${error instanceof Error ? error.message : String(error)}` });
    }
  };

  const isProviderMode = route === 'provider' && preset != null;

  if (loading) {
    return <div style={{ padding: 24, textAlign: 'center', color: 'var(--semi-color-text-2)' }}>{t('common.loading')}</div>;
  }

  if (step === 'choose') {
    return (
      <div>
        <button type="button" className="ga-form-back" onClick={onDone}>
          ← {t('common.cancel')}
        </button>

        <div className="ga-route-picker">
          <span className="ga-route-btn ga-route-btn--active" aria-current="true">
            {t('model.routeProvider')}
          </span>
          <button
            type="button"
            className="ga-route-btn"
            onClick={handleSelectCustom}
          >
            {t('model.routeCustom')}
          </button>
        </div>

        <div className="ga-provider-list">
          {PROVIDER_PRESETS.map((p) => (
            <ProviderCard
              key={p.key}
              preset={p}
              onClick={() => handleSelectProvider(p)}
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="ga-form">
      {!isEdit && (
        <button type="button" className="ga-form-back" onClick={handleBack}>
          ← {t('model.selectProvider')}
        </button>
      )}
      {isEdit && (
        <button type="button" className="ga-form-back" onClick={onDone}>
          ← {t('common.cancel')}
        </button>
      )}

      {/* Alias */}
      <div className="ga-form-field">
        <label className="ga-form-label">
          {t('model.name')}
        </label>
        <Input
          value={form.name}
          onChange={(val) => updateForm({ name: val })}
          placeholder={t('model.namePh')}
          maxLength={50}
        />
      </div>

      {/* Model ID */}
      <div className="ga-form-field">
        <label className="ga-form-label">
          {t('model.model')} <span className="ga-form-req">*</span>
        </label>
        <Input
          value={form.model}
          onChange={(val) => updateForm({ model: val })}
          placeholder={t('model.modelPh')}
          maxLength={50}
        />
      </div>

      {/* Protocol */}
      <div className="ga-form-field">
        <label className="ga-form-label">
          {t('model.protocol')} <span className="ga-form-req">*</span>
        </label>
        <RadioGroup
          value={form.protocol}
          onChange={(e) => updateForm({ protocol: e.target.value })}
          disabled={isProviderMode}
        >
          <Radio value="oai">{t('model.protocolOai')}</Radio>
          <Radio value="claude">{t('model.protocolClaude')}</Radio>
        </RadioGroup>
        {isProviderMode && (
          <span className="ga-form-locked-hint">{t('model.providerLocked')}</span>
        )}
      </div>

      {/* API URL */}
      <div className="ga-form-field">
        <label className="ga-form-label">
          {t('model.apibase')} <span className="ga-form-req">*</span>
        </label>
        <Input
          value={form.apibase}
          onChange={(val) => updateForm({ apibase: val })}
          placeholder={t('model.apibasePh')}
          maxLength={200}
          disabled={isProviderMode}
        />
        {isProviderMode && (
          <span className="ga-form-locked-hint">{t('model.providerLocked')}</span>
        )}
      </div>

      {/* API Key */}
      <div className="ga-form-field">
        <label className="ga-form-label">
          {t('model.apikey')} {!isEdit && <span className="ga-form-req">*</span>}
          {isProviderMode && preset && (
            <a
              className="ga-form-key-link"
              href={preset.keyUrl}
              target="_blank"
              rel="noopener noreferrer"
            >
              {t('model.getKey')}
            </a>
          )}
        </label>
        <Input
          mode="password"
          value={form.apikey}
          onChange={(val) => updateForm({ apikey: val })}
          placeholder={t('model.apikeyPh')}
          maxLength={200}
        />
      </div>

      {/* Stream */}
      <div className="ga-form-field">
        <label className="ga-form-label">{t('model.stream')}</label>
        <RadioGroup
          value={form.stream}
          onChange={(e) => updateForm({ stream: e.target.value })}
        >
          <Radio value={true}>{t('model.streamOn')}</Radio>
          <Radio value={false}>{t('model.streamOff')}</Radio>
        </RadioGroup>
      </div>

      {/* Advanced toggle */}
      <button
        type="button"
        className="ga-form-advanced-toggle"
        data-expanded={showAdvanced || undefined}
        onClick={() => setShowAdvanced((v) => !v)}
      >
        <svg width="10" height="10" viewBox="0 0 10 10" fill="currentColor">
          <path d="M3 1.5L7 5L3 8.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" fill="none" />
        </svg>
        {t('model.advanced')}
      </button>

      {showAdvanced && (
        <div className="ga-form-advanced-body">
          <div className="ga-form-row">
            <div className="ga-form-field">
              <label className="ga-form-label">{t('model.retries')}</label>
              <InputNumber
                value={form.max_retries}
                onChange={(val) => updateForm({ max_retries: val as number })}
                min={1}
                max={20}
                style={{ width: '100%' }}
              />
            </div>
            <div className="ga-form-field">
              <label className="ga-form-label">{t('model.connTimeout')}</label>
              <InputNumber
                value={form.connect_timeout}
                onChange={(val) => updateForm({ connect_timeout: val as number })}
                min={1}
                max={300}
                suffix="s"
                style={{ width: '100%' }}
              />
            </div>
            <div className="ga-form-field">
              <label className="ga-form-label">{t('model.readTimeout')}</label>
              <InputNumber
                value={form.read_timeout}
                onChange={(val) => updateForm({ read_timeout: val as number })}
                min={5}
                max={3600}
                suffix="s"
                style={{ width: '100%' }}
              />
            </div>
          </div>
        </div>
      )}

      {/* Probe + Submit */}
      <div className="ga-form-actions">
        <div className="ga-form-probe" data-state={probe.state === 'done' ? (probe.result.ok ? 'ok' : 'fail') : probe.state} role="status">
          {probe.state === 'running' && t('model.testing')}
          {probe.state === 'done' && (probe.result.ok
            ? t('model.testOk', { ms: probe.result.latencyMs ?? 0 })
            : `${t('model.testFail')}${probe.result.status ? ` (HTTP ${probe.result.status})` : ''}: ${probe.result.error ?? ''}`)}
        </div>
        <Button onClick={handleProbe} loading={probe.state === 'running'} disabled={!requiredFilled}>
          {t('model.test')}
        </Button>
        <Button type="primary" onClick={handleSubmit}>
          {t('model.save')}
        </Button>
      </div>
    </div>
  );
}
