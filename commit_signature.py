"""Opt-in commit signature: set GA_COMMIT_SIGNATURE (or use /signature) to append a trailer to commits."""
import os

_sig = os.environ.get('GA_COMMIT_SIGNATURE', '').strip()

def prompt_block():
    return f"When you create a git commit, append `{_sig}` as the final line of the commit message.\n" if _sig else ''

def toggle(arg=''):
    global _sig
    a = (arg or '').strip()
    if a.lower() in ('', 'status'): return f"commit signature: {_sig or 'off'}"
    _sig = '' if a.lower() == 'off' else (os.environ.get('GA_COMMIT_SIGNATURE', '').strip() if a.lower() == 'on' else a)
    return f"commit signature: {_sig or 'off'}"
