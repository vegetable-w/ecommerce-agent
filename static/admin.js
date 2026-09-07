/* 管理画面の共通シェル。
 *
 * - style は各ページに配らずここから注入する。既存ページ(index.html)は style を
 *   インラインで持っているので、ホスト側の指定を上書きしないよう .adm- 接頭辞で閉じる。
 * - 全体を IIFE で包む。index.html は $ / el / log といった短い名前をトップレベルで
 *   使っているため、素で定義すると将来ページを統合したときに衝突する。
 * - 配色・角丸・カプセル表現は 01 章のチャット画面に合わせる(ユーザー指示)。外部依存は入れない。
 */
(function () {
  const MODULES = [
    { key: 'admin', href: '/admin', label: 'ホーム' },
    { key: 'kb', href: '/kb', label: 'ナレッジベース' },
    { key: 'rageval', href: '/rag-eval', label: 'RAG 評価' },
    { key: 'chat', href: '/', label: 'チャット画面' },
  ];

  const CSS = `
  :root{
    --bg1:#fff1f5; --bg2:#eef2ff; --bg3:#fdf6ec;
    --ink:#3d3550; --muted:#9b91ad;
    --line:#f0e6f2; --card:#ffffff;
    --accent:#ff8fb1; --accent-d:#ff6f9c;
    --ok:#5ed6a4; --warn:#ffc46b; --bad:#ff8a8a; --none:#cfc6da;
  }
  *{box-sizing:border-box}
  body{
    margin:0; color:var(--ink);
    font-family:"Hiragino Maru Gothic ProN","Yu Gothic UI",-apple-system,"Segoe UI",system-ui,sans-serif;
    background:
      radial-gradient(1000px 600px at 10% -10%, var(--bg1), transparent 60%),
      radial-gradient(900px 500px at 110% 10%, var(--bg2), transparent 60%),
      var(--bg3);
    min-height:100vh;
  }
  .adm-wrap{max-width:1080px; margin:0 auto; padding:18px 20px 60px}

  .adm-nav{display:flex; align-items:center; gap:10px; padding:12px 16px; margin-bottom:18px;
    background:rgba(255,255,255,.82); backdrop-filter:blur(10px);
    border:1px solid #fff; border-radius:20px;
    box-shadow:0 10px 30px rgba(160,120,170,.16), 0 2px 0 #fff inset}
  .adm-nav .adm-face{width:34px;height:34px;border-radius:50%;flex:none;display:grid;place-items:center;
    font-size:18px; background:linear-gradient(160deg,#ffd9e5,#ffeaf1);
    box-shadow:0 3px 10px rgba(255,143,177,.35)}
  .adm-nav b{font-size:14.5px; margin-right:6px}
  .adm-nav a{
    font-size:13px; text-decoration:none; color:var(--muted);
    border:1px solid var(--line); background:#fff; padding:6px 13px; border-radius:999px;
    transition:color .15s, border-color .15s}
  .adm-nav a:hover{color:var(--accent-d); border-color:#ffd3e0}
  .adm-nav a.on{color:#fff; border-color:transparent;
    background:linear-gradient(160deg,var(--accent),var(--accent-d));
    box-shadow:0 4px 12px rgba(255,143,177,.42)}

  .adm-card{background:var(--card); border:1px solid var(--line); border-radius:20px;
    padding:16px 18px; margin-bottom:14px;
    box-shadow:0 6px 20px rgba(170,140,180,.09)}
  .adm-card > h2{margin:0 0 12px; font-size:15px; display:flex; align-items:center; gap:8px}
  .adm-card > h2 .adm-num{width:22px;height:22px;border-radius:50%;flex:none;display:grid;place-items:center;
    font-size:12px; color:#fff; background:linear-gradient(160deg,var(--accent),var(--accent-d))}

  .adm-bar{display:flex; flex-wrap:wrap; gap:9px; margin-bottom:14px}
  .adm-stat{flex:1; min-width:132px; background:#fff; border:1px solid var(--line);
    border-radius:16px; padding:10px 14px}
  .adm-stat .k{font-size:11.5px; color:var(--muted)}
  .adm-stat .v{font-size:21px; font-weight:600; line-height:1.35}
  .adm-stat .v.na{font-size:14px; color:var(--muted); font-weight:400}

  .adm-pill{display:inline-flex; align-items:center; gap:5px; font-size:11.5px;
    padding:3px 10px; border-radius:999px; border:1px solid var(--line); background:#fdfafc; color:var(--muted)}
  .adm-pill i{width:7px;height:7px;border-radius:50%;display:block;background:var(--none)}
  .adm-pill.ok i{background:var(--ok)} .adm-pill.warn i{background:var(--warn)}
  .adm-pill.bad i{background:var(--bad)}
  .adm-pill.dashed{border-style:dashed}

  .adm-btn{font:inherit; font-size:12.5px; cursor:pointer; border:1px solid var(--line);
    background:#fff; color:var(--ink); padding:7px 14px; border-radius:999px; transition:transform .1s}
  .adm-btn:hover:not(:disabled){border-color:#ffd3e0; color:var(--accent-d)}
  .adm-btn:active:not(:disabled){transform:translateY(1px)}
  .adm-btn:disabled{opacity:.45; cursor:not-allowed}
  .adm-btn.pri{border-color:transparent; color:#fff;
    background:linear-gradient(160deg,var(--accent),var(--accent-d));
    box-shadow:0 4px 12px rgba(255,143,177,.40)}
  .adm-btn.heavy{border-color:#ffd9d9; color:#b4444f; background:#fff6f6}
  .adm-btns{display:flex; flex-wrap:wrap; gap:8px}

  .adm-tbl{width:100%; border-collapse:collapse; font-size:13px}
  .adm-tbl th,.adm-tbl td{border-bottom:1px solid var(--line); padding:7px 9px; text-align:left; vertical-align:top}
  .adm-tbl th{font-weight:600; color:var(--muted); font-size:11.5px; white-space:nowrap}
  .adm-tbl tr:last-child td{border-bottom:none}
  .adm-scroll{overflow-x:auto}

  .adm-in,.adm-ta{font:inherit; font-size:13.5px; color:var(--ink); background:#fff;
    border:1.5px solid var(--line); border-radius:14px; padding:9px 13px; outline:none; width:100%}
  .adm-in:focus,.adm-ta:focus{border-color:#ffc7d8}
  .adm-ta{resize:vertical; min-height:120px; line-height:1.65; font-family:ui-monospace,Consolas,monospace; font-size:12.5px}
  .adm-row{display:flex; flex-wrap:wrap; gap:9px; align-items:center}
  select.adm-in{width:auto; min-width:130px}

  .adm-log{background:#2f2838; color:#f4eef7; border-radius:14px; padding:11px 13px; margin-top:10px;
    font-family:ui-monospace,Consolas,monospace; font-size:11.5px; line-height:1.65;
    white-space:pre-wrap; word-break:break-all; max-height:230px; overflow:auto}
  .adm-log:empty{display:none}

  .adm-note{font-size:12px; color:var(--muted); line-height:1.7; margin:6px 0 0}
  .adm-err{background:#fff4f4; border:1px solid #ffd9d9; color:#b4444f;
    border-radius:14px; padding:9px 13px; font-size:12.5px; margin-top:8px}
  .adm-empty{color:var(--muted); font-size:12.5px; padding:10px 2px}
  .adm-toast{position:fixed; left:50%; bottom:26px; transform:translateX(-50%);
    background:#fff; border:1px solid var(--line); border-radius:999px; padding:10px 20px;
    font-size:13px; box-shadow:0 10px 30px rgba(160,120,170,.28); z-index:50}
  `;

  function injectStyles() {
    if (document.getElementById('adm-style')) return;
    const s = document.createElement('style');
    s.id = 'adm-style';
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  /* 新しい管理画面を足すときは MODULES に 1 行足すだけ。各ページの URL は変えない。 */
  function mountAdminNav(active) {
    injectStyles();
    const nav = document.createElement('nav');
    nav.className = 'adm-nav';
    nav.innerHTML =
      '<div class="adm-face">🐱</div><b>STORE 管理画面</b>' +
      MODULES.map(m =>
        `<a href="${m.href}"${m.key === active ? ' class="on"' : ''}>${m.label}</a>`
      ).join('');
    const host = document.querySelector('.adm-wrap') || document.body;
    host.insertBefore(nav, host.firstChild);
  }

  async function getJSON(url, opts) {
    const res = await fetch(url, opts);
    let body = null;
    try { body = await res.json(); } catch { /* 本文が JSON でないこともある */ }
    if (!res.ok) {
      const msg = (body && (body.detail || body.message)) || `HTTP ${res.status}`;
      const err = new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
      err.status = res.status;
      throw err;
    }
    return body;
  }

  const postJSON = (url, data) => getJSON(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(data || {}),
  });

  /* 取得できない値は 0 ではなく「—」。0 件であることと読めないことは別物。 */
  function statBox(label, value) {
    const na = value === null || value === undefined;
    return `<div class="adm-stat"><div class="k">${label}</div>` +
           `<div class="v${na ? ' na' : ''}">${na ? '取得不可' : value}</div></div>`;
  }

  function pill(kind, text, dashed) {
    return `<span class="adm-pill ${kind}${dashed ? ' dashed' : ''}"><i></i>${text}</span>`;
  }

  function esc(t) {
    return String(t == null ? '' : t)
      .replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
  }

  let toastTimer = null;
  function toast(msg) {
    document.querySelectorAll('.adm-toast').forEach(n => n.remove());
    const d = document.createElement('div');
    d.className = 'adm-toast';
    d.textContent = msg;
    document.body.appendChild(d);
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => d.remove(), 4000);
  }

  window.Admin = { mountAdminNav, getJSON, postJSON, statBox, pill, esc, toast };
})();
