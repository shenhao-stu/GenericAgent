// @vitest-environment node
import { describe, it, expect } from 'vitest';
import { BUILTIN_SKILLS, matchSkillPrefix, skillDescription, skillTitle } from '../components/chat/Composer/skills';
import { zh } from '../i18n/zh';
import { en } from '../i18n/en';

describe('BUILTIN_SKILLS', () => {
  it('has unique ids', () => {
    const ids = BUILTIN_SKILLS.map((s) => s.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('all skills have non-empty prompts', () => {
    for (const skill of BUILTIN_SKILLS) {
      expect(skill.prompt.length).toBeGreaterThan(10);
    }
  });

  it('every built-in skill is titled and described in both languages through i18n', () => {
    for (const skill of BUILTIN_SKILLS) {
      for (const dict of [zh, en]) {
        const t = (key: string) => dict[key] ?? key;
        expect(skillTitle(skill, t), `${skill.id} title`).toBe(dict[`preset.${skill.id}.t`]);
        expect(skillDescription(skill, t, 'zh'), `${skill.id} description`).toBe(dict[`preset.${skill.id}.d`]);
        expect(dict[`preset.${skill.id}.t`]).toBeTruthy();
        expect(dict[`preset.${skill.id}.d`]).toBeTruthy();
      }
    }
  });

  it('custom presets fall back to their own title and description', () => {
    const custom = { id: 'weekly', title: 'Weekly report', desc: { zh: '周报', en: 'Weekly' }, prompt: 'write it' };
    const t = (key: string) => key;
    expect(skillTitle(custom, t)).toBe('Weekly report');
    expect(skillDescription(custom, t, 'zh')).toBe('周报');
    expect(skillDescription({ ...custom, desc: undefined }, t, 'zh')).toBe('');
  });
});

describe('matchSkillPrefix', () => {
  it('matches plan skill prompt', () => {
    const planSkill = BUILTIN_SKILLS.find((s) => s.id === 'plan')!;
    const result = matchSkillPrefix(planSkill.prompt + ' build a chat app');
    expect(result).not.toBeNull();
    expect(result!.id).toBe('plan');
    expect(result!.rest).toBe('build a chat app');
  });

  it('matches goal skill prompt with no rest', () => {
    const goalSkill = BUILTIN_SKILLS.find((s) => s.id === 'goal')!;
    const result = matchSkillPrefix(goalSkill.prompt);
    expect(result).not.toBeNull();
    expect(result!.id).toBe('goal');
    expect(result!.rest).toBe('');
  });

  it('returns null for non-matching content', () => {
    expect(matchSkillPrefix('Hello world')).toBeNull();
    expect(matchSkillPrefix('')).toBeNull();
  });

  it('returns null for partial prefix match', () => {
    const planSkill = BUILTIN_SKILLS.find((s) => s.id === 'plan')!;
    expect(matchSkillPrefix(planSkill.prompt.slice(0, 10))).toBeNull();
  });

  it('matches all built-in skills by their own prompt', () => {
    for (const skill of BUILTIN_SKILLS) {
      const result = matchSkillPrefix(skill.prompt + ' extra text');
      expect(result, `failed for skill "${skill.id}"`).not.toBeNull();
      expect(result!.id).toBe(skill.id);
    }
  });
});
