"""Commit signature: GA_COMMIT_SIGNATURE=true/false (or /signature on|off) toggles a fixed Co-Authored-By trailer."""
import os

_SIGNATURE = "Co-Authored-By: GenericAgent <bot@gaagent.ai>"
_on = os.environ.get('GA_COMMIT_SIGNATURE', 'false').strip().lower() in ('1', 'true', 'on', 'yes')

def prompt_block():
    return f"When you create a git commit, append `{_SIGNATURE}` as the final line of the commit message.\n" if _on else ''

def toggle(arg=''):
    global _on
    a = (arg or '').strip().lower()
    if a == 'on': _on = True
    elif a == 'off': _on = False
    return f"commit signature: {'on' if _on else 'off'}"
