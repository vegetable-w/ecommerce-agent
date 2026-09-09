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
    { key: 'review', href: '/review', label: '査読キュー' },
    { key: 'observability', href: '/observability', label: '可観測性' },
    { key: 'chat', href: '/', label: 'チャット画面' },
  ];

  const CSS = `
  /* 配色は中立のグレーに 1 色のアクセント。色は「意味」にだけ使い、装飾には使わない。
     運用者が数字と状態を読む画面なので、彩度で目を引くより、余白と字面の階層で読ませる。 */
  :root{
    --page:#f6f7f9;          /* 画面の地。カードとの差でカードを浮かせる */
    --card:#ffffff;
    --ink:#1c2024;           /* 本文。真っ黒にしない(眩しさと滲みを避ける) */
    --ink-2:#3f4650;         /* 見出し以外の濃いめ */
    --muted:#6b7480;         /* ラベル。AA を満たす明度に置く */
    --line:#e3e6ea;
    --line-2:#eef0f3;        /* 表の罫線。区切りは見えれば十分で、目立たせない */
    --accent:#2563eb;
    --accent-d:#1d4ed8;
    --accent-soft:#eff4ff;
    --ok:#0f9960; --warn:#b45309; --bad:#c0392b; --none:#98a1ac;
    --ok-bg:#e8f6ef; --warn-bg:#fdf3e3; --bad-bg:#fdeceb; --none-bg:#f2f4f6;
    --radius:10px;
    --radius-sm:7px;
  }
  *{box-sizing:border-box}
  body{
    margin:0; color:var(--ink);
    font-size:14px; line-height:1.65;
    font-family:-apple-system,"Segoe UI","Yu Gothic UI","Hiragino Sans",
                "Noto Sans JP",system-ui,sans-serif;
    background:var(--page);
    min-height:100vh;
    -webkit-font-smoothing:antialiased;
  }
  .adm-wrap{max-width:1180px; margin:0 auto; padding:0 24px 64px}

  /* ナビ。画面の一番上に貼り付けて、どこにいるかを常に見せる */
  .adm-nav{
    display:flex; align-items:center; gap:6px; flex-wrap:wrap;
    padding:0 24px; margin:0 -24px 24px; height:56px;
    background:var(--card); border-bottom:1px solid var(--line);
    position:sticky; top:0; z-index:20;
  }
  .adm-nav .adm-face{
    width:26px; height:26px; border-radius:6px; flex:none; display:grid; place-items:center;
    font-size:13px; font-weight:700; color:#fff; background:var(--ink);
  }
  .adm-nav b{font-size:14.5px; font-weight:600; margin:0 14px 0 4px; white-space:nowrap}
  .adm-nav a{
    font-size:13.5px; text-decoration:none; color:var(--muted);
    padding:6px 11px; border-radius:var(--radius-sm);
    transition:background .12s, color .12s;
  }
  .adm-nav a:hover{background:var(--page); color:var(--ink)}
  /* いま開いているページ。下線ではなく塗りにする(タブが折り返しても位置が崩れない) */
  .adm-nav a.on{color:var(--accent-d); background:var(--accent-soft); font-weight:600}

  .adm-card{
    background:var(--card); border:1px solid var(--line); border-radius:var(--radius);
    padding:20px 22px; margin-bottom:16px;
  }
  .adm-card > h2{
    margin:0 0 16px; font-size:15.5px; font-weight:600; letter-spacing:.01em;
    display:flex; align-items:center; gap:9px;
  }
  /* 手順の番号。丸より角丸の方が数字が中央に見え、桁が増えても崩れない */
  .adm-card > h2 .adm-num{
    width:22px; height:22px; border-radius:6px; flex:none; display:grid; place-items:center;
    font-size:12px; font-weight:600; color:var(--accent-d); background:var(--accent-soft);
  }

  .adm-bar{display:flex; flex-wrap:wrap; gap:12px; margin-bottom:18px}
  .adm-stat{
    flex:1; min-width:150px; background:var(--card);
    border:1px solid var(--line); border-radius:var(--radius); padding:12px 15px;
  }
  .adm-stat .k{font-size:12px; color:var(--muted); letter-spacing:.02em}
  /* 数字は等幅にする。並べたときに桁が揃わないと大小が読み取れない */
  .adm-stat .v{
    font-size:25px; font-weight:600; line-height:1.35; color:var(--ink);
    font-variant-numeric:tabular-nums;
  }
  .adm-stat .v.na{font-size:14px; color:var(--muted); font-weight:400}

  /* 状態のバッジ。色は 4 つだけで、それぞれ意味が決まっている */
  .adm-pill{
    display:inline-flex; align-items:center; gap:6px; font-size:12.5px; font-weight:500;
    padding:3px 10px; border-radius:999px;
    border:1px solid var(--line); background:var(--none-bg); color:var(--muted);
  }
  .adm-pill i{width:6px; height:6px; border-radius:50%; display:block; background:var(--none)}
  .adm-pill.ok{background:var(--ok-bg); border-color:#c7e8d8; color:var(--ok)}
  .adm-pill.ok i{background:var(--ok)}
  .adm-pill.warn{background:var(--warn-bg); border-color:#f0dcb4; color:var(--warn)}
  .adm-pill.warn i{background:var(--warn)}
  .adm-pill.bad{background:var(--bad-bg); border-color:#f5cdc9; color:var(--bad)}
  .adm-pill.bad i{background:var(--bad)}
  .adm-pill.dashed{border-style:dashed}

  .adm-btn{
    font:inherit; font-size:13.5px; font-weight:500; cursor:pointer;
    border:1px solid var(--line); background:var(--card); color:var(--ink-2);
    padding:7px 15px; border-radius:var(--radius-sm);
    transition:background .12s, border-color .12s, color .12s;
  }
  .adm-btn:hover:not(:disabled){background:var(--page); border-color:#d3d8de}
  .adm-btn:active:not(:disabled){background:#eceff2}
  .adm-btn:disabled{opacity:.5; cursor:not-allowed}
  /* キーボード操作でどこにいるか見えるようにする。マウスでは出さない */
  .adm-btn:focus-visible,.adm-in:focus-visible,.adm-ta:focus-visible,.adm-nav a:focus-visible{
    outline:2px solid var(--accent); outline-offset:2px;
  }
  .adm-btn.pri{border-color:var(--accent); background:var(--accent); color:#fff}
  .adm-btn.pri:hover:not(:disabled){background:var(--accent-d); border-color:var(--accent-d)}
  /* 課金するか、戻せない操作。押す前に色で気づかせる */
  .adm-btn.heavy{border-color:#f0cfcb; background:var(--bad-bg); color:var(--bad)}
  .adm-btn.heavy:hover:not(:disabled){background:#fbdedb}
  .adm-btns{display:flex; flex-wrap:wrap; gap:8px}

  .adm-tbl{width:100%; border-collapse:collapse; font-size:13.5px}
  .adm-tbl th,.adm-tbl td{
    border-bottom:1px solid var(--line-2); padding:10px 12px;
    text-align:left; vertical-align:top;
  }
  .adm-tbl th{
    font-weight:600; color:var(--muted); font-size:11.5px; white-space:nowrap;
    text-transform:uppercase; letter-spacing:.05em;
    border-bottom-color:var(--line);
  }
  .adm-tbl td{font-variant-numeric:tabular-nums}
  .adm-tbl tbody tr:hover td{background:#fafbfc}
  .adm-tbl tr:last-child td{border-bottom:none}
  .adm-scroll{overflow-x:auto}

  .adm-in,.adm-ta{
    font:inherit; font-size:13.5px; color:var(--ink); background:var(--card);
    border:1px solid var(--line); border-radius:var(--radius-sm);
    padding:8px 11px; outline:none; width:100%;
    transition:border-color .12s, box-shadow .12s;
  }
  .adm-in:focus,.adm-ta:focus{border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-soft)}
  .adm-in::placeholder,.adm-ta::placeholder{color:#9aa3ae}
  .adm-ta{
    resize:vertical; min-height:120px; line-height:1.7;
    font-family:ui-monospace,"SFMono-Regular",Consolas,monospace; font-size:12.5px;
  }
  .adm-row{display:flex; flex-wrap:wrap; gap:9px; align-items:center}
  select.adm-in{width:auto; min-width:130px}

  .adm-log{
    background:#11151a; color:#d7dde5; border:1px solid #1e242c;
    border-radius:var(--radius-sm); padding:12px 14px; margin-top:12px;
    font-family:ui-monospace,"SFMono-Regular",Consolas,monospace; font-size:12.5px; line-height:1.75;
    white-space:pre-wrap; word-break:break-all; max-height:240px; overflow:auto;
  }
  .adm-log:empty{display:none}

  .adm-note{font-size:13px; color:var(--muted); line-height:1.8; margin:8px 0 0}
  .adm-err{
    background:var(--bad-bg); border:1px solid #f5cdc9; color:var(--bad);
    border-radius:var(--radius-sm); padding:10px 14px; font-size:13.5px; margin-top:10px;
  }
  .adm-empty{color:var(--muted); font-size:13.5px; padding:16px 2px}
  .adm-toast{
    position:fixed; left:50%; bottom:28px; transform:translateX(-50%);
    background:var(--ink); color:#fff; border-radius:var(--radius-sm);
    padding:11px 18px; font-size:13.5px; max-width:min(560px,90vw);
    box-shadow:0 8px 24px rgba(16,20,26,.24); z-index:60;
  }
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
      '<div class="adm-face">S</div><b>STORE 管理コンソール</b>' +
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
