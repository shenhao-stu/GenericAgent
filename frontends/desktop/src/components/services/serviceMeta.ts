/** Display metadata for bridge-managed runtime components, keyed by the bridge's service id. */
export interface ServiceMeta {
  labelKey: string;
  summaryKey: string;
  tipKey: string;
}

export const SERVICE_META: Record<string, ServiceMeta> = {
  '__bridge__': { labelKey: 'proc.bridge', summaryKey: 'proc.bridgeSummary', tipKey: 'proc.bridgeTip' },
  'frontends/conductor.py': { labelKey: 'proc.conductor', summaryKey: 'proc.conductorSummary', tipKey: 'proc.conductorTip' },
  'reflect/scheduler.py': { labelKey: 'proc.scheduler', summaryKey: 'proc.schedulerSummary', tipKey: 'proc.schedulerTip' },
};

export const serviceLabelKey = (id: string): string => SERVICE_META[id]?.labelKey ?? 'proc.runtimeComponent';
