import os, sys, re, threading, asyncio, queue as Q, socket, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TEMP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'temp')
from agentmain import GeneraticAgent
try:
    from telegram import BotCommand
    from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
    from telegram.request import HTTPXRequest
except:
    print("Please ask the agent install python-telegram-bot to use telegram module.")
    sys.exit(1)
from chatapp_common import HELP_TEXT, TELEGRAM_MENU_COMMANDS, format_restore
from continue_cmd import handle_frontend_command, reset_conversation
from llmcore import mykeys

agent = GeneraticAgent()
agent.verbose = False
ALLOWED = set(mykeys.get('tg_allowed_users', []))

_TAG_PATS = [r'<' + t + r'>.*?</' + t + r'>' for t in ('thinking', 'summary', 'tool_use')]
_TAG_PATS.append(r'<file_content>.*?</file_content>')

def _clean(t):
    for p in _TAG_PATS:
        t = re.sub(p, '', t, flags=re.DOTALL)
    return re.sub(r'\n{3,}', '\n\n', t).strip() or '...'

import html as _html
def _inline_md(s):
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<i>\1</i>', s)
    s = re.sub(r'`([^`]+)`', r'<code>\1</code>', s)
    return s
def _to_html(t):
    parts, pos = [], 0
    for m in re.finditer(r'(`{3,})(?:\w*\n)?([\s\S]*?)\1', t):
        parts.append(_inline_md(_html.escape(t[pos:m.start()])))
        parts.append('<pre><code>' + _html.escape(m.group(2)) + '</code></pre>')
        pos = m.end()
    parts.append(_inline_md(_html.escape(t[pos:])))
    return ''.join(parts)

async def _stream(dq, msg):
    last_text = ""
    while True:
        await asyncio.sleep(3)
        item = None
        try:
            while True: item = dq.get_nowait()
        except Q.Empty: pass
        if item is None: continue
        raw = item.get("done") or item.get("next", "")
        done = "done" in item
        show = _clean(raw)
        if len(show) > 4000:
            # freeze current msg, start a new one
            try: msg = await msg.reply_text("(continued...)")
            except Exception: pass
            last_text = ""
            show = show[-3900:]
        display = show if done else show + " ⏳"
        if display != last_text:
            try: await msg.edit_text(_to_html(display), parse_mode='HTML')
            except Exception:
                try: await msg.edit_text(display)
                except Exception: pass
            last_text = display
        if done:
            files = re.findall(r'\[FILE:([^\]]+)\]', show[-1000:])
            for fpath in files:
                if not os.path.isabs(fpath): fpath = os.path.join(_TEMP_DIR, fpath)
                if os.path.exists(fpath):
                    if fpath.lower().endswith(('.png','.jpg','.jpeg','.gif','.webp')):
                        try: await msg.reply_photo(open(fpath,'rb'))
                        except Exception: pass
                    else:
                        try: await msg.reply_document(open(fpath,'rb'))
                        except Exception: pass
            show = re.sub(r'\[FILE:[^\]]+\]', '', show)
            if show.strip():
                try: await msg.edit_text(_to_html(show), parse_mode='HTML')
                except Exception:
                    try: await msg.edit_text(show)
                    except Exception: pass
            break

def _normalized_command(text):
    parts = (text or '').strip().split(None, 1)
    if not parts:
        return ''
    head = parts[0].lower()
    if head.startswith('/'):
        head = '/' + head[1:].split('@', 1)[0]
    return head + (f" {parts[1].strip()}" if len(parts) > 1 and parts[1].strip() else '')

def _cancel_stream_task(ctx):
    task = ctx.user_data.pop('stream_task', None)
    if task and not task.done():
        task.cancel()

async def _sync_commands(application):
    await application.bot.set_my_commands([BotCommand(command, description) for command, description in TELEGRAM_MENU_COMMANDS])

async def handle_msg(update, ctx):
    uid = update.effective_user.id
    if ALLOWED and uid not in ALLOWED:
        return await update.message.reply_text("no")
    msg = await update.message.reply_text("thinking...")
    prompt = f"If you need to show files to user, use [FILE:filepath] in your response.\n\n{update.message.text}"
    dq = agent.put_task(prompt, source="telegram")
    task = asyncio.create_task(_stream(dq, msg))
    ctx.user_data['stream_task'] = task

async def cmd_abort(update, ctx):
    _cancel_stream_task(ctx)
    agent.abort()
    await update.message.reply_text("⏹️ 正在停止...")

async def cmd_llm(update, ctx):
    args = (update.message.text or '').split()
    if len(args) > 1:
        try:
            n = int(args[1])
            agent.next_llm(n)
            await update.message.reply_text(f"✅ 已切换到 [{agent.llm_no}] {agent.get_llm_name()}")
        except (ValueError, IndexError):
            await update.message.reply_text(f"用法: /llm <0-{len(agent.list_llms())-1}>")
    else:
        lines = [f"{'→' if cur else '  '} [{i}] {name}" for i, name, cur in agent.list_llms()]
        await update.message.reply_text("LLMs:\n" + "\n".join(lines))

async def handle_photo(update, ctx):
    uid = update.effective_user.id
    if ALLOWED and uid not in ALLOWED:
        return await update.message.reply_text("no")
    photo = update.message.photo[-1]
    file = await photo.get_file()
    fpath = f"tg_{photo.file_unique_id}.jpg"
    await file.download_to_drive(os.path.join(_TEMP_DIR, fpath))
    caption = update.message.caption
    prompt = f"[TIPS] 收到图片temp/{fpath}\n{caption}" if caption else f"[TIPS] 收到图片temp/{fpath}，请等待下一步指令"
    msg = await update.message.reply_text("thinking...")
    dq = agent.put_task(prompt, source="telegram")
    task = asyncio.create_task(_stream(dq, msg))
    ctx.user_data['stream_task'] = task

async def handle_command(update, ctx):
    uid = update.effective_user.id
    if ALLOWED and uid not in ALLOWED:
        return await update.message.reply_text("no")
    cmd = _normalized_command(update.message.text)
    op = cmd.split()[0] if cmd else ''
    if op == '/help': return await update.message.reply_text(HELP_TEXT)
    if op == '/status':
        llm = agent.get_llm_name() if agent.llmclient else '未配置'
        return await update.message.reply_text(f"状态: {'🔴 运行中' if agent.is_running else '🟢 空闲'}\nLLM: [{agent.llm_no}] {llm}")
    if op == '/stop': return await cmd_abort(update, ctx)
    if op == '/llm': return await cmd_llm(update, ctx)
    if op == '/new':
        _cancel_stream_task(ctx)
        return await update.message.reply_text(reset_conversation(agent))
    if op == '/restore':
        _cancel_stream_task(ctx)
        try:
            restored_info, err = format_restore()
            if err:
                return await update.message.reply_text(err)
            restored, fname, count = restored_info
            agent.abort()
            agent.history.extend(restored)
            return await update.message.reply_text(f"✅ 已恢复 {count} 轮对话\n来源: {fname}\n(仅恢复上下文，请输入新问题继续)")
        except Exception as e:
            return await update.message.reply_text(f"❌ 恢复失败: {e}")
    if op == '/continue':
        if cmd != '/continue': _cancel_stream_task(ctx)
        return await update.message.reply_text(handle_frontend_command(agent, cmd))
    return await update.message.reply_text(HELP_TEXT)

if __name__ == '__main__':
    try:  # Single instance lock using socket
        _lock_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); _lock_sock.bind(('127.0.0.1', 19527))
    except OSError: 
        print('[Telegram] Another instance is already running, skiping...')
        sys.exit(1)
    if not ALLOWED: 
        print('[Telegram] ERROR: tg_allowed_users in mykey.py is empty or missing. Set it to avoid unauthorized access.')
        sys.exit(1)
    _logf = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'temp', 'tgapp.log'), 'a', encoding='utf-8', buffering=1)
    sys.stdout = sys.stderr = _logf
    print('[NEW] New process starting, the above are history infos ...')
    threading.Thread(target=agent.run, daemon=True).start()
    proxy = mykeys.get('proxy', 'http://127.0.0.1:2082')
    print('proxy:', proxy)

    async def _error_handler(update, context: ContextTypes.DEFAULT_TYPE):
        print(f"[{time.strftime('%m-%d %H:%M')}] TG error: {context.error}", flush=True)

    while True:
        try:
            print(f"TG bot starting... {time.strftime('%m-%d %H:%M')}")
            # Recreate request and app objects on each restart to avoid stale connections
            request = HTTPXRequest(proxy=proxy, read_timeout=30, write_timeout=30, connect_timeout=30, pool_timeout=30)
            app = (ApplicationBuilder().token(mykeys['tg_bot_token'])
                   .request(request).get_updates_request(request).post_init(_sync_commands).build())
            app.add_handler(MessageHandler(filters.COMMAND, handle_command))
            app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
            app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_msg))
            app.add_error_handler(_error_handler)
            app.run_polling(drop_pending_updates=True, poll_interval=1.0, timeout=30)
        except Exception as e:
            print(f"[{time.strftime('%m-%d %H:%M')}] polling crashed: {e}", flush=True)
            time.sleep(10)
            asyncio.set_event_loop(asyncio.new_event_loop())
