export interface SkillDef {
  id: string;
  title: string;
  /** Localized description; custom presets may omit it. */
  desc?: { zh: string; en: string };
  prompt: string;
}

export const BUILTIN_SKILLS: SkillDef[] = [
  {
    id: 'plan',
    title: 'Plan',
    prompt: 'Enter Plan mode: read memory/plan_sop.md, follow Explore → Plan → Execute → Verify flow for the task I describe next.',
  },
  {
    id: 'goal',
    title: 'Goal',
    prompt: 'Enter Goal mode: read L3 goal mode SOP, autonomously achieve the goal I describe next.',
  },
  {
    id: 'autonomous',
    title: 'Autonomous',
    prompt: 'Enter autonomous mode: read memory/autonomous_operation_sop.md, select or plan tasks, execute independently and produce a report.',
  },
  {
    id: 'hive',
    title: 'Hive',
    prompt: 'Start Goal Hive mode: follow hive SOP, spawn multiple workers to collaboratively achieve my next goal.',
  },
  {
    id: 'review',
    title: 'Review',
    prompt: 'Enter reviewer mode: rigorously critique the latest output, check each item and report issues.',
  },
  {
    id: 'findwork',
    title: 'Find Work',
    prompt: 'Analyze my situation using the autonomous planning approach, generate a batch of TODOs that would interest me.',
  },
];

type TFn = (key: string) => string;

/** Built-in skills are titled/described through i18n (`preset.<id>.t/.d`); custom presets keep their own text. */
export function skillTitle(skill: SkillDef, t: TFn): string {
  const key = `preset.${skill.id}.t`;
  const text = t(key);
  return text === key ? skill.title : text;
}

export function skillDescription(skill: SkillDef, t: TFn, lang: 'zh' | 'en'): string {
  const key = `preset.${skill.id}.d`;
  const text = t(key);
  if (text !== key) return text;
  return skill.desc?.[lang] || skill.desc?.en || '';
}

export function matchSkillPrefix(content: string): { id: string; rest: string } | null {
  for (const skill of BUILTIN_SKILLS) {
    if (content.startsWith(skill.prompt)) {
      const rest = content.slice(skill.prompt.length).trimStart();
      return { id: skill.id, rest };
    }
  }
  return null;
}
