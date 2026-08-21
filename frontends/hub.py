"""GA Hub: `import hub` -> client (silent if no server); `python hub.py` -> WS server.
    hub.connect(agent, 'stapp')     # or override any hook: connect(a, n, put_task=, get_outputs=, abort=)
    default put -> 'busy' if agent.is_running, else parks plain text in agent._hub_inbox for the UI
HTTP (errors = {'error','code'} + status: offline/gone 404, busy 409, timeout 504, nosupport 501, badop 400):
    GET peers -> [{name,title,n_msgs,sig}] | {name}/messages?detail=1&sig= -> {title,tasks:[{i,input,steps:
    [{j,title,n}]}],sig} or {same:1,sig} | {name}/seg/{i}/{j}?off=N -> {content,off,n} (step bodies, tailable)
    POST {name}/put {"text":..} -> {ok:1} | {name}/abort -> {ok:1}
    POST sense/ingest {"items":[obs..],"cursor":..} -> {ok,accepted,dup,bad,next_cursor} (phone archive, agent-free)
    GET  sense/query?sel={json} -> {ok,items,total,evidence,abstain_hint} | POST sense/purge {sel|{"all":1}} -> {ok,deleted}
         sel = 手机端 sense.py 那一份(from/to/as_of/kinds/sid/ids/with/q/near{place|lat,lon,radius_m,places}/include_invalid); 别的键一律 400
    POST sense/blob {id,seq,total,b64,sha256} -> {ok,have,total} (opt-in audio chunks; one file per chunk, joined when complete)
"""
import os, re, sys, json, time, asyncio, threading, random, hmac, hashlib
PORT = int(os.environ.get('GA_HUB_PORT', 19736))
WEB_PORT = int(os.environ.get('GA_HUB_WEB_PORT', PORT + 1))   # the only face fit to be tunnelled out (frp this one)
TOKEN = os.environ.get('GA_HUB_TOKEN') or ''.join(random.choices('abcdefghijkmnpqrstuvwxyz23456789', k=6))
URL = os.environ.get('GA_HUB', f'ws://127.0.0.1:{PORT}/ws')
TITLE_MIN = 12                    # a too short last input gets the previous one prepended
NOISE = re.compile(r'\**LLM Running \(Turn \d+\) \.\.\.\**|`{3,}.*?`{3,}|<thinking>.*?</thinking>', re.DOTALL)

class HubClient:
    def __init__(self, name, put_task=None, get_outputs=None, abort=None, state=None, on_ev=None, sub=(), fixed=False):
        self.name, self.put_task, self.get_outputs, self.abort = name, put_task, get_outputs, abort
        self.state, self.on_ev, self.sub, self.fixed = state or dict, on_ev, list(sub), fixed  # fixed -> stable name
        self._tc, self._ws, self._lp = {}, None, None   # (i,j) -> (fp, title): a finished step is never re-scanned
        self._stop = False                              # set by stop(): the reconnect loop is the only thing that reads it
    def emit(self, topic, data=None, to=None):          # thread-safe fire-and-forget from the host thread
        if self._ws and self._lp: asyncio.run_coroutine_threadsafe(self._ws.send(json.dumps(
            {'op': 'ev', 'topic': topic, 'to': to, 'data': data}, default=str)), self._lp)
    def start(self):
        threading.Thread(target=lambda: asyncio.run(self._loop()), daemon=True).start(); return self
    def stop(self):
        """Leave the bus for good. A host with a dynamic set of peers (one per session/tab) needs this:
        without it a closed session lingers as a ghost row in /api/peers forever, reconnecting every 5s."""
        self._stop = True
        ws, lp = self._ws, self._lp
        if ws and lp:
            try: asyncio.run_coroutine_threadsafe(ws.close(), lp)   # drops us from `peers` on the server's finally
            except Exception: pass
    async def _loop(self):
        try: import websockets
        except ImportError: return
        while not self._stop:
            try:
                async with websockets.connect(URL, open_timeout=3, max_size=None) as ws:
                    self._ws, self._lp = ws, asyncio.get_running_loop()
                    if self._stop: self._ws = None; break   # stop() raced the connect: leave now, not as a ghost peer
                    caps = [k for k, v in (('get', self.get_outputs), ('put', self.put_task), ('abort', self.abort)) if v]
                    await ws.send(json.dumps({'op': 'hello', 'name': self.name, 'pid': os.getpid(),
                                              'fixed': self.fixed, 'caps': caps, 'sub': self.sub}))
                    async for raw in ws: await self._on_cmd(ws, json.loads(raw))
            except Exception: pass         # silent: the hub is optional, it must not disturb the host
            self._ws = None
            if self._stop: break
            await asyncio.sleep(5)
    def _stitle(self, i, j, s):
        s, e = s or '', self._tc.get((i, j))
        fp = (len(s), s[:24] + s[-24:])    # memoised on length+edges: guards index reuse after /clear
        if e and e[0] == fp: return e[1]
        body = NOISE.sub('', s)            # title = <summary> if present, else first meaningful line
        m = re.search(r'<summary>\s*(.*?)\s*</summary>', body, re.DOTALL)
        t = (m.group(1) if m else body.strip()).strip().split('\n')[0]; t = t[:50] + '...' if len(t) > 50 else t or f'step {j + 1}'
        self._tc[(i, j)] = (fp, t); return t
    def _build(self, c):
        """Skeleton only (no bodies), off-loop: regex/serialisation must not stall the WS reader."""
        tasks = list(self.get_outputs() or []); outs = [list(t.get('outputs') or []) for t in tasks]  # copy: host mutates
        sig = '.'.join(str(len(o)) for o in outs) + ':' + str(len(outs[-1][-1] or '') if outs and outs[-1] else 0)
        if c.get('sig') == sig: return {'same': 1, 'sig': sig, **self.state()}   # nothing moved -> no payload at all
        ins = [(t.get('input') or '').strip() for t in tasks]
        ins = [u for u in ins if u and not u.startswith('/')] or [u for u in ins if u]   # /clear is plumbing
        title = ins[-1].split('\n')[0] if ins else '(empty)'
        w = sum(2 if ord(ch) > 0x2E80 else 1 for ch in title)   # display width: a CJK char is worth two
        if len(ins) > 1 and w < TITLE_MIN: title = ins[-2].split('\n')[0] + ' <- ' + title
        det, rows = int(c.get('detail', 1)), []
        since = max(0, int(c.get('since') or 0))   # delta: the client already holds rows < since (they are immutable)
        for i, (t, o) in enumerate(zip(tasks, outs)):
            if i < since: continue
            steps = [{'j': j, 'title': self._stitle(i, j, s), 'n': len(s or '')} for j, s in enumerate(o)]
            rows.append({'i': i, 'input': t.get('input', ''), **({'steps': steps} if det else {'ns': len(o)})})
        return {'title': title[:80], 'tasks': rows, 'nt': len(tasks), 'sig': sig, **self.state()}   # nt lets a delta client spot /clear
    def _seg(self, c):
        tasks, off = list(self.get_outputs() or []), max(0, int(c.get('off') or 0))
        i, j = int(c.get('i', -1)), int(c.get('j', -1))
        o = (tasks[i].get('outputs') or []) if -len(tasks) <= i < len(tasks) else None
        if o is None or not (-len(o) <= j < len(o)):
            return {'error': f'segment {i}/{j} gone (trimmed or cleared)', 'code': 'gone'}
        s = o[j] or ''; return {'content': s[off:], 'off': off, 'n': len(s)}          # off lets a UI tail a growing step
    async def _on_cmd(self, ws, c):
        op = c.get('op'); data = {'error': f'unknown op {op}', 'code': 'badop'}
        if op == 'ev': return self.on_ev and self.on_ev(c)      # bus push: no reply, must not block the reader
        if op in ('get', 'seg'): data = await asyncio.to_thread(self._build if op == 'get' else self._seg, c)
        elif op == 'put_task': data = (c.get('text') and await asyncio.to_thread(self.put_task, c['text'])) or {'ok': 1}
        elif op == 'abort': data = (await asyncio.to_thread(self.abort) or {'ok': 1}) if self.abort else {'error': 'no abort hook', 'code': 'nosupport'}
        await ws.send(json.dumps({'op': 'r', 'id': c.get('id'), 'name': self.name, 'data': data}, default=str))

def serve():
    """Bring the hub up on demand: any host may spawn it, the port is the lock (a loser just exits).
    Detached + windowless, so it outlives its spawner and no console ever flashes."""
    import socket, subprocess
    if '127.0.0.1' not in URL and 'localhost' not in URL: return    # a remote hub is not ours to start
    with socket.socket() as s:
        if s.connect_ex(('127.0.0.1', PORT)) == 0: return           # already listening
    exe = os.path.join(os.path.dirname(sys.executable), 'pythonw.exe')
    subprocess.Popen([exe if os.path.exists(exe) else sys.executable, os.path.abspath(__file__)],
                     cwd=os.path.dirname(os.path.abspath(__file__)), close_fds=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                     creationflags=0x08000008 if os.name == 'nt' else 0)   # DETACHED | NO_WINDOW

def connect(agent, name=None, put_task=None, get_outputs=None, abort=None, fold=None):
    """One line to wire a GA host: `hub.connect(agent, 'stapp')`; any hook can still be overridden.
    Default put refuses while the agent is busy (a remote must not cut in line), else it parks
    plain text in agent._hub_inbox; the UI feeds it through its own input entrance when idle,
    so remote tasks behave exactly like typed ones (bubble/stream/stop). Requires a live UI tab.
    Never raises: a broken hub must not break its host."""
    def _put(text):
        if getattr(agent, 'is_running', False): return {'error': f'peer {name} is busy', 'code': 'busy'}
        agent._hub_inbox.append(text)   # no put_task here: the UI's unified entrance sends it
    try:
        try: serve()                                   # best effort: bring up a local hub if none is listening
        except Exception: pass
        if not hasattr(agent, '_hub_inbox'): agent._hub_inbox = []
        agent._hub = HubClient(name or getattr(agent, 'name', 'agent'), put_task or _put,
                               get_outputs or (lambda: agent.all_outputs), abort or agent.abort,
                               state=lambda: {'run': bool(getattr(agent, 'is_running', False))})
        return agent._hub.start()
    except Exception: return None

# ------------------------------- server (python hub.py) -------------------------------
if __name__ == '__main__':
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response
    from fastapi.responses import FileResponse, JSONResponse; import uvicorn
    HERE, OK_ORIGIN = os.path.dirname(os.path.abspath(__file__)), ('', f'http://127.0.0.1:{PORT}', f'http://localhost:{PORT}')
    STATUS = {'offline': 404, 'gone': 404, 'timeout': 504, 'nosupport': 501, 'badop': 400, 'busy': 409}
    app, peers, meta, waits, seq, pcache, pdead = FastAPI(), {}, {}, {}, [0], {}, {}  # ws / hello / futures / row / probe
    @app.middleware('http')
    async def guard(req: Request, call_next):
        if req.scope['server'][1] == PORT: return Response(status_code=404)   # bus port speaks WebSocket only
        public = getattr(app.state, 'p2p_pair_open', False) and req.url.path in ('/pair', '/pair/status')
        if public: return await call_next(req)
        t = req.query_params.get('t') or req.cookies.get('ga_hub_t') or ''    # web port: token or nothing
        if not hmac.compare_digest(t, TOKEN): return Response(status_code=403)
        r = await call_next(req); r.set_cookie('ga_hub_t', TOKEN, httponly=True, samesite='lax'); return r
    async def ask(name, msg, timeout=15):
        if name not in peers: return {'error': f'peer {name} offline', 'code': 'offline'}
        seq[0] += 1; rid = seq[0]
        waits[rid] = fut = asyncio.get_running_loop().create_future()
        try:
            await peers[name].send_text(json.dumps(dict(msg, id=rid)))
            return await asyncio.wait_for(fut, timeout)    # the WS reader resolves it
        except asyncio.TimeoutError: return {'error': f'peer busy (>{timeout}s)', 'code': 'timeout'}
        except Exception as e: return {'error': str(e) or type(e).__name__, 'code': 'offline'}
        finally: waits.pop(rid, None)
    async def fan(m, frm):      # the bus: unicast when 'to' is set, else every peer subscribed to the topic
        for n, w in list(peers.items()):
            if n != frm and (m.get('to') == n or (not m.get('to') and m.get('topic') in (meta.get(n, {}).get('sub') or []))):
                try: await w.send_text(json.dumps({'op': 'ev', 'topic': m.get('topic'), 'from': frm, 'data': m.get('data')}))
                except Exception: pass
    def out(d):                 # a peer-level error becomes a real HTTP status, not a 200 with a body
        c = d.get('code') if isinstance(d, dict) else None; return JSONResponse(d, STATUS.get(c, 400)) if c else d
    @app.websocket('/ws')
    async def endpoint(ws: WebSocket):
        if ws.scope['server'][1] != PORT or (ws.headers.get('origin') or '') not in OK_ORIGIN:
            return await ws.close(code=1008)      # the bus lives on PORT alone and never leaves this machine
        await ws.accept(); name = None
        try:
            while True:
                m = json.loads(await ws.receive_text())
                if m.get('op') == 'hello':      # fixed -> one stable, addressable name (pet, notifier, scheduler...)
                    name = re.sub(r'[^\w.-]', '_', m['name'] if m.get('fixed') else f"{m.get('name')}-{m.get('pid')}")
                    m['caps'] = m.get('caps', ['get', 'put', 'abort'])   # legacy clients (no caps in hello) = full agent
                    peers[name], meta[name] = ws, m; print(f'[+] {name} {m["caps"]} ({len(peers)} online)')
                elif m.get('op') == 'ev': await fan(m, name)
                f = waits.get(m.get('id')) if m.get('op') == 'r' else None
                if f and not f.done(): f.set_result(m.get('data'))
        except WebSocketDisconnect: pass
        finally:
            if name and peers.get(name) is ws:      # never evict the successor that replaced us
                peers.pop(name, None); meta.pop(name, None); print(f'[-] {name} ({len(peers)} online)')
    @app.get('/api/peers')
    async def api_peers(psig: str = None):   # psig given -> {same:1} while the row set holds (no psig = full list)
        now = time.time()
        names = [n for n in peers if 'get' in (meta.get(n, {}).get('caps') or []) and now >= pdead.get(n, 0)]
        rs = await asyncio.gather(*[ask(n, {'op': 'get', 'detail': 0,
                                            'sig': (pcache.get(n) or {}).get('sig')}, 3) for n in names])
        for n, r in zip(names, rs): pdead[n] = now + 15 if (r or {}).get('error') else 0
        rs, rows = dict(zip(names, rs)), []
        for n in peers:
            r, caps = rs.get(n), meta.get(n, {}).get('caps') or []
            if 'get' not in caps: rows.append({'name': n, 'title': '', 'n_msgs': 0, 'sig': None, 'caps': caps}); continue
            if r is None or r.get('error') or (r.get('same') and n in pcache):   # busy/skipped/idle -> last row
                rows.append({**(pcache.get(n) or {'name': n, 'title': '?', 'n_msgs': 0, 'sig': None}),
                             'caps': caps, 'run': (r or {}).get('run')}); continue
            msgs = sum(1 for t in r.get('tasks', []) if not (t.get('input') or '').lstrip().startswith('/'))
            rows.append({'name': n, 'title': r.get('title', '?'), 'n_msgs': msgs, 'sig': r.get('sig'),
                         'caps': caps, 'run': r.get('run')})
            if r.get('sig'): pcache[n] = rows[-1]
        for n in [x for x in pcache if x not in peers]: pcache.pop(n, None)
        if psig is None: return rows
        sg = hashlib.md5(json.dumps(rows, sort_keys=True).encode()).hexdigest()[:12]
        return {'same': 1} if psig == sg else {'rows': rows, 'psig': sg}
    @app.get('/api/{name}/messages')
    async def api_messages(name: str, detail: int = 1, sig: str = None, since: int = 0): return out(await ask(name, {'op': 'get', 'detail': detail, 'sig': sig, 'since': since}))
    @app.get('/api/{name}/seg/{i}/{j}')
    async def api_seg(name: str, i: int, j: int, off: int = 0): return out(await ask(name, {'op': 'seg', 'i': i, 'j': j, 'off': off}))
    put_seen = {}   # ikey -> (ts, outcome|None): put 幂等窗(r15)。手机端对同 peer 同文案的重试沿用同一 ikey;
    # WS send 一旦发生就算「已投递」(peer 多半已执行, 只是应答超了 15s 窗), 同键重放绝不能再投一次 ——
    # 这正是「手机新建会话, PC 冒出两个」的重试路径。未离开 hub 的失败(offline/gone/busy/badop/nosupport)放行重试。
    @app.post('/api/{name}/put')
    async def api_put(name: str, body: dict):
        ik = str(body.get('ikey') or '')
        now = time.time()
        if ik:
            for k in [k for k, (ts, _) in put_seen.items() if now - ts > 300]: put_seen.pop(k, None)
            if ik in put_seen: return put_seen[ik][1] or {'ok': 1, 'dedup': 1}
            put_seen[ik] = (now, None)                      # sent 标记先落: 并发同键第二发立即走上面的重放分支
        r = await ask(name, {'op': 'put_task', 'text': body.get('text', '')})
        code = r.get('code') if isinstance(r, dict) else None
        if ik:
            if code in ('offline', 'gone', 'busy', 'badop', 'nosupport'): put_seen.pop(ik, None)   # 没投出去, 允许重试
            else: put_seen[ik] = (time.time(), None if code else r)  # timeout 也封键(已进 peer 的 WS); 封窗从应答时刻起算, 不吃掉 ask 的等待
        return out(r)
    @app.post('/api/{name}/abort')
    async def api_abort(name: str): return out(await ask(name, {'op': 'abort'}))
    # ---- optional P2P pairing; delete this block to remove it completely ----
    try:
        from hub_p2p import install as _install_p2p
        _install_p2p(app, web_port=WEB_PORT, token=TOKEN, here=HERE)
    except Exception:
        pass  # missing plug-in/client/dependency must not affect the hub

    # ---- optional phone->PC sense archive; delete this block to remove it completely ----
    # 手机 /sense.sync 经 P2P 打到这里: hub_p2p 的 allow=('/api/',) 是前缀白名单, 新 /api/* 天生手机端可达(端侧零改动)。
    # 纯落盘、绝不经 agent: PC 没开 agent 时也不能丢数据。日档 JSONL + 内存 id 集去重, 结构即索引。
    try:
        import math, base64
        SDIR, MAXB = os.path.join(HERE, 'sense_store'), 8 * 1024 * 1024   # == HTTPExporter.max_body: 提前拒, 好过传一半被通道砍断
        _seen, _red, _slk, _sready = set(), set(), threading.Lock(), []   # _red: 只存在性上来的空壳, 是唯一允许被顶掉的一档
        def _sfiles(): return sorted(f for f in os.listdir(SDIR) if f.endswith('.jsonl')) if os.path.isdir(SDIR) else []
        def _sread(f):
            try: fh = open(os.path.join(SDIR, f), encoding='utf-8')
            except OSError: return
            with fh:
                for ln in fh:
                    try: yield json.loads(ln)
                    except Exception: pass          # 断电写半行只丢该行, 不能让整天的档案不可读
        def _sinit():
            if _sready: return
            with _slk:
                for f in _sfiles():
                    for o in _sread(f):
                        _seen.add(o.get('id'))
                        if o.get('redacted'): _red.add(o.get('id'))
                try: _seen.update(open(os.path.join(SDIR, 'tomb.idx'), encoding='utf-8').read().split())
                except OSError: pass                # 墓碑: 已删的 id 永不复活(迟到的重推按 dup 丢弃), 隐私优先于完整
                _sready.append(1)
        # ↓ selector 的语义只有一份, 权威在手机端 sense.py 的 _prep/_match/_near_hit, 这里是它的逐键移植。
        #   【两处必须同步改】: PC 多认一个键就多删, 少认一个键就漏删, 而回执里只有 deleted:N, 用户无从分辨。
        SEL = ('from', 'to', 'as_of', 'kinds', 'sid', 'ids', 'with', 'q', 'near', 'include_invalid')
        def _sbad(s):                               # 看不懂的键只有一种安全处理: 拒。忽略它 = 按剩下的条件放大删除范围
            return [k for k in s if k not in SEL + ('limit', 'all', 'selector')]
        def _sprep(s):                              # 对齐 _prep: 一次归一, 之后 _hit 保持纯函数
            n = s.get('near') if isinstance(s.get('near'), dict) else None
            near = None
            if n and n.get('place'):
                near = {'places': [str(n['place'])]}
            elif n and n.get('lat') is not None and n.get('lon') is not None:
                r = n.get('radius_m') if n.get('radius_m') is not None else n.get('r_m')   # 文档键是 radius_m; r_m 是 _prep 同样接受的别名
                near = {'lat': float(n['lat']), 'lon': float(n['lon']), 'r_m': float(r or 300) or 300.0,
                        'places': [str(p) for p in (n.get('places') or [])]}   # 圈内 place 白名单只能端侧算(PC 没有 entity 表)
            return dict(s, near=near, q=(str(s['q']).lower() if s.get('q') else None),
                        ids=set(s['ids']) if s.get('ids') else None)
        def _near(o, n):
            g = o.get('geo') or {}
            if o.get('place_id') and o['place_id'] in n['places']: return True
            if n.get('lat') is None or g.get('lat') is None: return False   # 只有 place 的行, 靠上面那份白名单命中
            dx = ((g.get('lon') or 0) - n['lon']) * 111320 * math.cos(math.radians(n['lat'])); dy = (g['lat'] - n['lat']) * 110540
            return math.hypot(dx, dy) <= n['r_m']
        def _hit(o, s):                             # s 必须来自 _sprep; 判定顺序与手机端 _match 一一对应
            a, b = o.get('ts_start') or 0, o.get('ts_end') or o.get('ts_start') or 0
            if s.get('from') and b < s['from']: return False
            if s.get('to') and a > s['to']: return False
            if s.get('as_of') and (o.get('ts_ingest') or 0) > s['as_of']: return False   # 时间因果性: 只看当时已入库的
            if s.get('kinds') and o.get('kind') not in s['kinds']: return False
            if s.get('sid') and o.get('sid') != s['sid']: return False
            if s.get('ids') and o.get('id') not in s['ids']: return False
            if s.get('with') and not set(s['with']) <= set(o.get('peers') or []): return False   # AND(与手机端同口径): OR 会让「和张三李四一起」多召回, purge 级联时更会多删
            if s.get('q') and s['q'] not in ((o.get('text') or '')
                                             + json.dumps(o.get('payload') or {}, ensure_ascii=False)).lower():
                return False                        # q 只在正文里找: dump 整条会拿 id/kind/source 去撞, q='phone' 能命中整个档案
            if s.get('near') and not _near(o, s['near']): return False
            return bool(s.get('include_invalid')) or o.get('valid', True)
        def _gaps(s):                               # PC 只拥有已同步的部分: 没档的日子必须显式成 gap, 否则"没同步"会被读成"没发生"
            # reason 只写 no_archive: PC 分不清未同步还是已删, 端侧才知道 not_capturing/silenced/purged —— 不猜就是不编
            a, b, out = s.get('from'), s.get('to'), []
            have, t = {f[:8] for f in _sfiles()}, a
            while a and b and t < b:
                d = time.strftime('%Y%m%d', time.localtime(t / 1000))
                nxt = int(time.mktime(time.strptime(d, '%Y%m%d')) + 86400) * 1000   # 按本地日切齐(档案本就按本地日分文件), 否则缺口会糊到有数据的那天
                if d not in have:
                    if out and out[-1]['to'] >= t: out[-1]['to'] = min(nxt, b)
                    else: out.append({'from': t, 'to': min(nxt, b), 'reason': 'no_archive'})
                t = nxt
            return out
        def _store(raw):
            try: b = json.loads(raw or b'{}')
            except Exception: return JSONResponse({'ok': 0, 'error': 'bad json'}, 400)
            b = b if isinstance(b, dict) else {'items': b}
            src = [o for o in (b.get('items') or []) if isinstance(o, dict)]
            bad = len(b.get('items') or []) - len(src)      # 非法条目如实回报, 不静默吞
            _sinit(); acc, dup, buf, pend, upg = 0, 0, {}, set(), {}
            with _slk:
                for o in src:
                    oid = o.setdefault('id', 'obs_' + hashlib.md5(json.dumps(o, sort_keys=True, default=str).encode()).hexdigest()[:16])
                    day = time.strftime('%Y%m%d', time.localtime((o.get('ts_start') or o.get('ts_ingest') or time.time() * 1000) / 1000))
                    if oid in _seen or oid in pend:
                        # 去重的唯一例外: 当初 consent 未授予, 端侧只上行了存在性(text/payload/blob_ref 全空 + redacted),
                        # 用户后来授予了并把原文推上来。空壳不让位, 「同意后自动补齐」在 PC 上就永远不成立。
                        # 墓碑命中不在此列(_red 只由真实档案填, purge 时同步清掉): 已删不复活的优先级更高。
                        if oid in _red and not o.get('redacted'): upg.setdefault(day, {})[oid] = o; _red.discard(oid); acc += 1
                        else: dup += 1
                        continue
                    ids, lines = buf.setdefault(day, ([], []))
                    ids.append(oid); lines.append(json.dumps(o, ensure_ascii=False, default=str)); pend.add(oid); acc += 1
                    if o.get('redacted'): _red.add(oid)
                os.makedirs(SDIR, exist_ok=True)
                for day, (ids, lines) in buf.items():
                    with open(os.path.join(SDIR, day + '.jsonl'), 'a', encoding='utf-8') as fh: fh.write('\n'.join(lines) + '\n')
                    _seen.update(ids)               # 逐天提交: 中途失败时已落盘那天不会被重推写重复
                for day, m in upg.items():          # 就地换行而不是追加: 追加会让同一个 id 在检索里出现两次
                    p = os.path.join(SDIR, day + '.jsonl')
                    rows = [m.pop(o.get('id'), o) for o in _sread(day + '.jsonl')] + list(m.values())
                    with open(p + '.tmp', 'w', encoding='utf-8') as fh:
                        fh.write(''.join(json.dumps(o, ensure_ascii=False, default=str) + '\n' for o in rows))
                    os.replace(p + '.tmp', p)       # 原子替换: 崩在中途也不留半个档案
            return {'ok': 1, 'accepted': acc, 'dup': dup, 'bad': bad,
                    'next_cursor': str(b.get('cursor') or max([o.get('ts_ingest') or 0 for o in src] or [0]))}
        @app.post('/api/sense/ingest')
        async def sense_ingest(req: Request):
            too_big = JSONResponse({'ok': 0, 'error': 'body over 8MB', 'code': 'toobig'}, 413)
            if int(req.headers.get('content-length') or 0) > MAXB: return too_big
            raw = await req.body()
            return too_big if len(raw) > MAXB else await asyncio.to_thread(_store, raw)
        @app.get('/api/sense/query')
        def sense_query(sel: str = '{}'):           # sync def -> FastAPI 自动丢线程池, 全表扫描不阻塞事件循环
            try: s0 = json.loads(sel or '{}') or {}
            except Exception: return JSONResponse({'ok': 0, 'error': 'sel must be json'}, 400)
            if _sbad(s0):                           # 端侧问了我们看不懂的条件: 少回一条也比多回一条好, 直说
                return JSONResponse({'ok': 0, 'error': 'unknown selector keys: ' + ','.join(_sbad(s0)), 'code': 'badop'}, 400)
            _sinit(); s = _sprep(s0)
            rows = sorted((o for f in _sfiles() for o in _sread(f) if _hit(o, s)), key=lambda o: o.get('ts_start') or 0)
            gaps = _gaps(s)
            return {'ok': 1, 'items': rows[:max(1, min(int(s0.get('limit') or 50), 500))], 'total': len(rows), 'filters_applied': s0,
                    'evidence': {'n_hits': len(rows), 'gaps': gaps,
                                 'span_covered_ms': (rows[-1].get('ts_start') or 0) - (rows[0].get('ts_start') or 0) if rows else 0},
                    'abstain_hint': not rows or bool(gaps)}
        @app.post('/api/sense/purge')
        def sense_purge(body: dict):
            s0, wipe = (body.get('selector') if isinstance(body.get('selector'), dict) else body), bool(body.get('all'))
            if _sbad(s0):                           # 删除不可逆: 看不懂的键既不能忽略(等于放宽范围)也不能猜, 只能拒
                return JSONResponse({'ok': 0, 'error': 'unknown selector keys: ' + ','.join(_sbad(s0)), 'code': 'badop'}, 400)
            if not (wipe or any(s0.get(k) for k in SEL if k != 'include_invalid')):
                return JSONResponse({'ok': 0, 'error': 'refuse empty selector (pass all:1 to wipe)', 'code': 'badop'}, 400)
            s = dict(_sprep(s0), include_invalid=True)   # 与手机端 _h_data 同口径: 删除面向全部行, 无效行(墓碑/坏分片)也删
            _sinit(); n, nf = 0, 0
            with _slk:
                for f in _sfiles():
                    keep, gone = [], []
                    for o in _sread(f): (gone if wipe or _hit(o, s) else keep).append(o)
                    if not gone: continue
                    n += len(gone); nf += 1; p = os.path.join(SDIR, f)
                    _red.difference_update(o.get('id') for o in gone)   # 已删的 id 退出「可被顶掉」的一档, 免得重推复活
                    with open(os.path.join(SDIR, 'tomb.idx'), 'a', encoding='utf-8') as fh:
                        fh.write(''.join(o['id'] + '\n' for o in gone if o.get('id')))   # 墓碑先落: 先删档再崩会让已删数据被重推复活
                    if not keep: os.remove(p); continue
                    with open(p + '.tmp', 'w', encoding='utf-8') as fh:
                        fh.write(''.join(json.dumps(o, ensure_ascii=False) + '\n' for o in keep))
                    os.replace(p + '.tmp', p)       # 原子替换: 崩在中途也不留半个档案
            return {'ok': 1, 'deleted': n, 'files': nf}
        @app.post('/api/sense/blob')
        def sense_blob(body: dict):                 # 音频默认不出设备; 只有用户显式改了 uplink 才会有分片打到这里
            bid = ''.join(c for c in str(body.get('id') or '') if c.isalnum() or c in '_-')[:64]
            seq, total = int(body.get('seq') or 0), int(body.get('total') or 0)
            if not bid or seq < 0 or total <= 0: return JSONResponse({'ok': 0, 'error': 'need id/seq/total', 'code': 'badop'}, 400)
            raw = base64.b64decode(body.get('b64') or '')
            if body.get('sha256') and hashlib.sha256(raw).hexdigest() != body['sha256']:
                return JSONResponse({'ok': 0, 'error': 'sha256 mismatch', 'code': 'badchunk'}, 400)   # 坏片只丢该片, 端侧按 next_seq 重发
            d = os.path.join(SDIR, 'blobs'); os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, '%s.%06d' % (bid, seq)), 'wb') as fh: fh.write(raw)   # 一片一文件 ⇒ 重发天然幂等
            parts = sorted(f for f in os.listdir(d) if f.startswith(bid + '.') and f[-6:].isdigit())
            if len(parts) >= total:                 # 收齐才合成: 半段音频比没有更误导
                with open(os.path.join(d, bid + '.pcm'), 'wb') as fh:
                    for p in parts: fh.write(open(os.path.join(d, p), 'rb').read())
                for p in parts: os.remove(os.path.join(d, p))
            return {'ok': 1, 'id': bid, 'have': min(len(parts), total), 'total': total}
    except Exception:
        pass  # 落档是可选能力, 缺依赖/权限不得影响 hub 主体

    @app.get('/')
    async def index(): return FileResponse(os.path.join(HERE, 'hub.html'))
    @app.get('/vendor/{f}')
    async def vendor(f: str):
        p = os.path.join(HERE, 'desktop', 'static', 'vendor', os.path.basename(f))
        return FileResponse(p) if os.path.exists(p) else Response(status_code=404)
    def serve(p):
        uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=p, log_level='warning')).run()
    print(f'[hub] bus ws://127.0.0.1:{PORT}/ws (no HTTP)  |  panel http://127.0.0.1:{WEB_PORT}/?t={TOKEN}')
    threading.Thread(target=serve, args=(WEB_PORT,), daemon=True).start()
    try: serve(PORT)
    except OSError: sys.exit(f'hub already running on {PORT}')
