import json, threading, time, uuid, queue, socket, requests, traceback
from typing import Any
from simple_websocket_server import WebSocketServer, WebSocket
import bottle
from bottle import request

def safe_print(*a, **k):
    try: print(*a, **k)
    except: pass

class Session:
    def __init__(self, session_id, info, client=None, client_id=None, tab_id=None):
        self.id = session_id
        # Browser-side tab id; differs from self.id only when two profiles collide on a tab id.
        self.tab_id = str(tab_id) if tab_id is not None else session_id
        self.info = info
        self.connect_at = time.time()
        self.disconnect_at = None
        self.type = info.get('type', 'ws')
        self.ws_client = client if self.type in ('ws', 'ext_ws') else None
        self.http_queue = client if self.type == 'http' else None
        # Stable per Chrome-profile extension identity (see tmwd_cdp_bridge clientId).
        self.client_id = client_id or info.get('client_id')
    @property
    def url(self): return self.info.get('url', '')
    def is_active(self):
        if self.type == 'http' and time.time() - self.connect_at > 60: self.mark_disconnected()
        return self.disconnect_at is None
    def reconnect(self, client, info, client_id=None):
        self.info = info
        self.type = info.get('type', 'ws')
        if self.type in ('ws', 'ext_ws'):
            self.ws_client = client
            self.http_queue = None
        elif self.type == 'http':
            self.http_queue = client
        if client_id is not None:
            self.client_id = client_id
        elif info.get('client_id') is not None:
            self.client_id = info.get('client_id')
        self.connect_at = time.time()
        self.disconnect_at = None
    def mark_disconnected(self):
        if self.disconnect_at is None: safe_print(f"Tab disconnected: {self.url} (Session: {self.id})")
        self.disconnect_at = time.time()


def apply_ext_tabs_update(sessions, ext_clients, ws_client, client_id, tabs, *, log=safe_print):
    """Apply one extension's tab snapshot without touching other profiles.

    Pure-ish helper (mutates sessions/ext_clients) so multi-profile ownership can be unit-tested.

    Rules:
    1. Identify the sender by stable ``client_id`` (fallback: anon:<id(ws)>).
    2. If the same client_id reconnects on a new WebSocket, transfer ownership first.
    3. Prune ONLY tabs owned by this client_id that are missing from the snapshot.
    4. Never steal an active tab owned by a different client_id: tab ids are unique
       per browser only, so a colliding tab is registered under ``client_id:tab_id``
       instead of being dropped.
    """
    client_id = client_id or f"anon:{id(ws_client)}"
    prev = ext_clients.get(client_id)
    if prev is not None and prev is not ws_client:
        # SW restart / reconnect: move this profile's sessions onto the new socket.
        for sess in sessions.values():
            if sess.type == 'ext_ws' and sess.client_id == client_id:
                sess.ws_client = ws_client
        log(f"ext client {client_id} transferred to new WS {getattr(ws_client, 'address', ws_client)}")
    ext_clients[client_id] = ws_client

    current_tab_ids = {str(tab.get('id')) for tab in tabs if tab.get('id') is not None}
    log(f"tabs update client={client_id} tabs={current_tab_ids}")

    for sess in list(sessions.values()):
        if sess.type != 'ext_ws':
            continue
        if sess.client_id != client_id:
            continue
        if sess.tab_id not in current_tab_ids:
            sess.mark_disconnected()

    claimed, fresh, namespaced = [], [], []
    for tab in tabs:
        if tab.get('id') is None:
            continue
        tab_id = str(tab['id'])
        info = {
            'url': tab.get('url'),
            'title': tab.get('title', ''),
            'connected_at': time.time(),
            'type': 'ext_ws',
            'client_id': client_id,
        }
        sid = tab_id
        ns_key = f"{client_id}:{tab_id}"
        if ns_key in sessions:
            # Once namespaced, this client keeps its own key — never touch the plain one.
            sid = ns_key
        else:
            sess = sessions.get(sid)
            if sess and sess.is_active() and sess.client_id and sess.client_id != client_id:
                # Tab-id collision across profiles: give this one its own namespaced key.
                sid = ns_key
                namespaced.append(sid)
                log(f"tab {tab_id} owned by other client, registered as {sid}")
        sess = sessions.get(sid)
        if sess and sess.is_active():
            sess.info = info
            sess.ws_client = ws_client
            sess.client_id = client_id
        elif sess is None:
            sessions[sid] = Session(sid, info, ws_client, client_id=client_id, tab_id=tab_id)
            fresh.append(sid)
            log(f"New tab connected: {info.get('url')} (Session: {sid}, client: {client_id})")
        else:
            sess.reconnect(ws_client, info, client_id=client_id)
            fresh.append(sid)
            log(f"Tab reconnected: {info.get('url')} (Session: {sid}, client: {client_id})")
        claimed.append(sid)
    return {'client_id': client_id, 'claimed': claimed, 'fresh': fresh,
            'namespaced': namespaced, 'current': list(current_tab_ids)}


class TMWebDriver:  
    def __init__(self, host: str = '127.0.0.1', port: int = 18765):  
        self.host, self.port = host, port
        self.sessions, self.results, self.acks = {}, {}, {}
        # client_id -> current WebSocket for that Chrome-profile extension instance
        self.ext_clients = {}
        self.default_session_id = None  
        self.latest_session_id = None  
        self.is_remote = socket.socket().connect_ex((host, port+1)) == 0
        if not self.is_remote:  
            self.start_ws_server()  
            self.start_http_server()
        else:
            self.remote = f'http://{self.host}:{self.port+1}/link'

    def start_http_server(self):
        self.app = app = bottle.Bottle()
        @app.hook('before_request')
        def reject_web_origin(): request.headers.get('Origin') is not None and bottle.abort(403)

        @app.route('/api/longpoll', method=['GET', 'POST'])
        def long_poll():
            data = request.json
            session_id = data.get('sessionId')  
            session_info = {'url': data.get('url'), 'title': data.get('title', ''), 'type': 'http'}  
            if session_id not in self.sessions: 
                session = Session(session_id, session_info, queue.Queue())
                safe_print(f"Browser http connected: {session.url} (Session: {session_id})")  
                self.sessions[session_id] = session
            session = self.sessions[session_id]
            if session.disconnect_at is not None and session.type != 'http': session.reconnect(queue.Queue(), session_info)
            session.disconnect_at = None
            if session.type == 'http': msgQ = session.http_queue
            else: return json.dumps({"id": "", "ret": "use ws"})
            session.connect_at = start_time = time.time()
            while time.time() - start_time < 5:
                try:
                    msg = msgQ.get(timeout=0.2)
                    try: self.acks[json.loads(msg).get('id','')] = True
                    except Exception: traceback.print_exc()
                    return msg
                except queue.Empty: continue
            return json.dumps({"id": "", "ret": "next long-poll"})

        @app.route('/api/result', method=['GET','POST'])
        def result():
            data = request.json
            if data.get('type') == 'result':  
                self.results[data.get('id')] = {'success': True, 'data': data.get('result'), 'newTabs': data.get('newTabs', [])}  
            elif data.get('type') == 'error':  
                self.results[data.get('id')] = {'success': False, 'data': data.get('error'), 'newTabs': data.get('newTabs', [])}  
            return 'ok'

        @app.route('/link', method=['GET','POST'])
        def link():
            data = request.json
            if data.get('cmd') == 'get_all_sessions': return json.dumps({'r': self.get_all_sessions()}, ensure_ascii=False)  
            if data.get('cmd') == 'find_session': 
                url_pattern = data.get('url_pattern', '')
                return json.dumps({'r': self.find_session(url_pattern)}, ensure_ascii=False)
            if data.get('cmd') == 'execute_js':
                session_id = data.get('sessionId')
                code = data.get('code')
                timeout = float(data.get('timeout', 10.0))
                try: result = self.execute_js(code, timeout=timeout, session_id=session_id)
                except Exception as e: return json.dumps({'r': {'error': str(e)}}, ensure_ascii=False)
                try: safe_print('[remote result]', (str(code)[:50] + ' RESULT:' +str(result)[:50]).replace('\n', ' '))
                except Exception: pass
                return json.dumps({'r': result}, ensure_ascii=False)
            return 'ok'
        def run():
            from wsgiref.simple_server import make_server, WSGIServer, WSGIRequestHandler
            from socketserver import ThreadingMixIn
            class _T(ThreadingMixIn, WSGIServer): pass
            class _H(WSGIRequestHandler):
                def log_request(self, *a): pass
            make_server(self.host, self.port+1, app, server_class=_T, handler_class=_H).serve_forever()
        http_thread = threading.Thread(target=run, daemon=True)
        http_thread.start()  

    def clean_sessions(self):
        sids = list(self.sessions.keys())
        for sid in sids:
            session = self.sessions[sid]
            if not session.is_active() and time.time() - session.disconnect_at > 600:
                del self.sessions[sid]
    
    def start_ws_server(self) -> None:  
        driver = self  
        class JSExecutor(WebSocket):  
            def handle(self) -> None:  
                try:  
                    data = json.loads(self.data)  
                    if data.get('type') == 'ready':  
                        session_id = data.get('sessionId')  
                        session_info = {'url': data.get('url'), 'title': data.get('title', ''),
                            'connected_at': time.time(), 'type': 'ws'}  
                        driver._register_client(session_id, self, session_info)  
                    elif data.get('type') in ['ext_ready', 'tabs_update']:
                        result = apply_ext_tabs_update(
                            driver.sessions,
                            driver.ext_clients,
                            self,
                            data.get('clientId') or data.get('client_id'),
                            data.get('tabs') or [],
                        )
                        driver._apply_tabs_result(result)
                    elif data.get('type') == 'ping':
                        try: self.send_message('{"type":"pong"}')
                        except Exception: pass
                    elif data.get('type') == 'ack': driver.acks[data.get('id','')] = True
                    elif data.get('type') == 'result':  
                        driver.results[data.get('id')] = {'success': True, 'data': data.get('result'), 'newTabs': data.get('newTabs', [])}  
                    elif data.get('type') == 'error':  
                        driver.results[data.get('id')] = {'success': False, 'data': data.get('error'), 'newTabs': data.get('newTabs', [])}  
                except Exception as e:  
                    safe_print(f"Error handling message: {e}")  
                    if hasattr(self, 'data'): safe_print(self.data)  
            def connected(self): (f"New connection from {self.address}")  
            def handle_close(self): 
                safe_print(f"WS Connection closed: {self.address}")
                driver._unregister_client(self)  
        
        self.server = WebSocketServer(self.host, self.port, JSExecutor)  
        server_thread = threading.Thread(target=self.server.serve_forever)  
        server_thread.daemon = True  
        server_thread.start()  
        safe_print(f"WebSocket server running on ws://{self.host}:{self.port}")  
    
    def _apply_tabs_result(self, result) -> None:
        # Only a new/reconnected tab moves the implicit default;
        # a refresh snapshot must not hijack it from another profile.
        if result['fresh']:
            self.latest_session_id = result['fresh'][-1]
        if self.default_session_id is None and result['claimed']:
            self.default_session_id = result['claimed'][0]

    def _register_client(self, session_id: str, client: WebSocket, session_info, client_id=None) -> None:
        is_new_session = session_id not in self.sessions
        client_id = client_id or session_info.get('client_id')

        if is_new_session:
            session = Session(session_id, session_info, client, client_id=client_id)
            self.sessions[session_id] = session            
            safe_print(f"New tab connected: {session.url} (Session: {session_id}, client: {client_id})")  
        else:
            session = self.sessions[session_id]
            # Never let a different Chrome profile steal an active tab id.
            if (
                session.is_active()
                and session.client_id
                and client_id
                and session.client_id != client_id
            ):
                safe_print(
                    f"refuse register steal tab {session_id}: "
                    f"owned by {session.client_id}, claimed by {client_id}"
                )
                return
            session.reconnect(client, session_info, client_id=client_id)
            safe_print(f"Tab reconnected: {session.url} (Session: {session_id}, client: {client_id})")  

        self.latest_session_id = session_id
        if self.default_session_id is None: self.default_session_id = session_id 
    
    def _unregister_client(self, client: WebSocket) -> None:
        dead_clients = [cid for cid, ws in self.ext_clients.items() if ws is client]
        for cid in dead_clients:
            del self.ext_clients[cid]
            safe_print(f"ext client gone: {cid}")
        for session in self.sessions.values():
            if session.ws_client is client:
                session.mark_disconnected()
    
    def execute_js(self, code, timeout=15, session_id=None) -> Any:  
        if session_id is None: session_id = self.default_session_id  
        if self.is_remote:
            safe_print('remote_execute_js')
            response = self._remote_cmd({"cmd": "execute_js", "sessionId": session_id, 
                                         "code": code, "timeout": str(timeout)}).get('r', {})
            if response.get('error'): raise Exception(response['error'])
            return response
 
        session = self.sessions.get(session_id)
        if not session or not session.is_active(): 
            time.sleep(3)
            session = self.sessions.get(session_id)
            if not session or not session.is_active(): 
                alive_sessions = [s for s in self.sessions.values() if s.is_active()]
                if alive_sessions:
                    session = alive_sessions[0]  
                    safe_print(f"会话 {session_id} 未连接，自动切换到最新活动会话: {session.id}")
                    session_id = self.default_session_id = session.id
                if not session or not session.is_active(): 
                    raise ValueError(f"会话ID {session_id} 未连接")  

        tp = session.type
        if tp not in ('ws', 'http', 'ext_ws'):
            raise ValueError(f"Unsupported session type: {tp}")
        exec_id = str(uuid.uuid4())  
        payload_dict = {'id': exec_id, 'code': code}
        if tp == 'ext_ws': payload_dict['tabId'] = int(session.tab_id)
        payload = json.dumps(payload_dict)

        if tp in ['ws', 'ext_ws']: session.ws_client.send_message(payload)  
        elif tp == 'http': session.http_queue.put(payload)

        start_time = time.time()  
        self.clean_sessions() 
        hasjump = acked = False

        while exec_id not in self.results:  
            time.sleep(0.2)  
            if not acked and exec_id in self.acks:
                acked = True; start_time = time.time()
            if tp in ['ws', 'ext_ws']:
                if not session.is_active(): hasjump = True
                if hasjump and session.is_active():
                    return {'result': f"Session {session_id} reloaded.", "closed":1}
            if time.time() - start_time > timeout:  
                if tp in ['ws', 'ext_ws']:
                    if hasjump: return {'result': f"Session {session_id} reloaded and new page is loading...", 'closed':1}
                    if acked: return {"result": f"No response data in {timeout}s (ACK received, script may still be running)"}
                    return {"result": f"No response data in {timeout}s (no ACK, script may not have been delivered)"}
                elif tp == 'http':
                    if acked: return {"result": f"Session {session_id} no response in {timeout}s (delivered but no result)"}
                    return {"result": f"Session {session_id} no response in {timeout}s (script not polled)"}
        
        result = self.results.pop(exec_id)  
        if exec_id in self.acks: self.acks.pop(exec_id)
        if not result['success']: raise Exception(result['data'])  
        rr = {'data': result['data']}
        newtabs = result.get('newTabs', []); [x.pop('ts', None) for x in newtabs]
        if newtabs: rr['newTabs'] = newtabs
        return rr
    
    def _remote_cmd(self, cmd):
        try: return requests.post(self.remote, headers={"Content-Type": "application/json"}, json=cmd, timeout=30).json()
        except (ConnectionError, requests.exceptions.ConnectionError):
            raise ConnectionError("TMWebDriver master未运行，看tmwebdriver_sop后台启动一个TMWebDriver")

    def get_all_sessions(self):  
        if self.is_remote:
            return self._remote_cmd({"cmd": "get_all_sessions"}).get('r', [])
        out = []
        for session in self.sessions.values():
            if not session.is_active():
                continue
            item = {'id': session.id, **session.info}
            if session.client_id:
                item['client_id'] = session.client_id
            if session.type == 'ext_ws':
                item['tab_id'] = session.tab_id
            out.append(item)
        return out

    def get_session_dict(self):
        return {session['id']: session['url'] for session in self.get_all_sessions()}
        
    def find_session(self, url_pattern: str):
        if url_pattern == '': 
            session = self.sessions.get(self.latest_session_id)
            return [(session.id, session.info)] if session else []
        matching_sessions = []  
        for session in self.sessions.values():
            if not session.is_active(): continue
            if 'url' in session.info and url_pattern in session.info['url']:  
                matching_sessions.append((session.id, session.info))  
        return matching_sessions

    def set_session(self, url_pattern: str) -> bool:  
        if self.is_remote:
            matched = self._remote_cmd({"cmd": "find_session", "url_pattern": url_pattern}).get('r', [])
        else:
            matched = self.find_session(url_pattern)
        if not matched: return safe_print(f"警告: 未找到URL包含 '{url_pattern}' 的会话")  
        if len(matched) > 1: safe_print(f"警告: 找到多个URL包含 '{url_pattern}' 的会话，选择第一个")  
        self.default_session_id, info = matched[0]
        safe_print(f"成功设置默认会话: {self.default_session_id}: {info['url']}")  
        return self.default_session_id  
    
    def jump(self, url, timeout=10): self.execute_js(f"window.location.href='{url}'", timeout=timeout)
    
if __name__ == "__main__":
    driver = TMWebDriver(host='127.0.0.1', port=18765)