// @vitest-environment happy-dom
import { forwardRef, useImperativeHandle } from 'react';
import { describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Composer } from '../components/chat/Composer';


vi.mock('../components/chat/Composer/RichEditorInput', () => ({
  RichEditorInput: forwardRef((_props: unknown, ref) => {
    useImperativeHandle(ref, () => ({
      clear: vi.fn(),
      focus: vi.fn(),
      getText: () => '',
      setText: vi.fn(),
      setSkillChip: vi.fn(),
      insertChip: vi.fn(),
    }));
    return <div data-testid="editor" />;
  }),
}));
vi.mock('../components/chat/Composer/CompletionDrawer', () => ({ CompletionDrawer: () => null }));
vi.mock('../components/chat/Composer/AtRefPopover', () => ({ AtRefPopover: () => null }));
vi.mock('../components/chat/Composer/ContextMenu', () => ({ ContextMenu: () => null }));
vi.mock('../components/chat/Composer/AttachmentStrip', () => ({ AttachmentStrip: () => null }));
vi.mock('../components/chat/Composer/SkillPanel', () => ({ SkillPanel: () => null }));
vi.mock('../components/chat/Composer/PrimaryCTA', () => ({
  PrimaryCTA: () => null,
  computeCTAState: () => 'idle',
}));
vi.mock('../components/chat/Composer/StatusStack', () => ({ StatusStack: () => null }));
vi.mock('../components/chat/Composer/ModelSelector', () => ({
  ModelSelector: () => <span data-testid="session-model-control">Session model</span>,
}));

class ResizeObserverStub {
  observe() {}
  disconnect() {}
}
vi.stubGlobal('ResizeObserver', ResizeObserverStub);


describe('Composer context slots', () => {
  it('renders a supplied channel model control instead of the Session selector', () => {
    render(
      <Composer
        placeholder=""
        onSend={vi.fn()}
        onStop={vi.fn()}
        isGenerating={false}
        modelControl={<span data-testid="conductor-model-control">Conductor model</span>}
      />,
    );

    expect(screen.getByTestId('conductor-model-control')).toBeTruthy();
    expect(screen.queryByTestId('session-model-control')).toBeNull();
  });

  it('renders the leading slot only when the host supplies one', () => {
    const { rerender } = render(
      <Composer placeholder="" onSend={vi.fn()} onStop={vi.fn()} isGenerating={false} leading={<span data-testid="workdir" />} />,
    );
    expect(screen.getByTestId('workdir')).toBeTruthy();

    rerender(<Composer placeholder="" onSend={vi.fn()} onStop={vi.fn()} isGenerating={false} />);
    expect(screen.queryByTestId('workdir')).toBeNull();
  });
});
