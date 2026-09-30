/* Paper summaries — reading app.
 *
 * Reads papers/topics.json, builds a two-level left nav (topic -> papers), and
 * renders a paper's markdown on demand. No build step and no framework: the
 * summaries are the artefact, this only displays them.
 *
 * Comments are saved through papers/app/server.py to
 * papers/comments/<paper>.json, so a note survives the browser and can be read
 * without the app.
 *
 * The roving-focus keyboard handling follows history/summaries_html_v1.html.
 */

const NAV = document.getElementById('nav');
const VIEW = document.getElementById('view');
const SHELL = document.querySelector('.shell');
const FOLD_BTN = document.getElementById('fold');
const DATA_URL = '../topics.json';   // relative to papers/app/
const PAPERS_ROOT = '..';            // papers/
const FOLD_KEY = 'papers.railFolded';

let topics = [];
let openTopic = null;
let current = null;                  // { topic, paper } being read
let comments = [];

/* ------------------------------------------------------------------ helpers */

function el(tag, props = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
    else if (k === 'html') n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) n.appendChild(c);
  return n;
}

function msg(text, detail) {
  VIEW.replaceChildren(
    el('p', { class: 'msg' }, [
      el('span', { text }),
      detail ? el('code', { text: ' ' + detail }) : null,
    ])
  );
}

/* Sort by year, then first author — the order stated in the papers rule. */
function ordered(papers) {
  return papers.slice().sort((a, b) => a.year - b.year || a.authors.localeCompare(b.authors));
}

function firstAuthor(authors) {
  return String(authors || '').split(',')[0];
}

/* ------------------------------------------------------- folding the rail */

function applyFold(folded) {
  SHELL.classList.toggle('folded', folded);
  FOLD_BTN.setAttribute('aria-expanded', String(!folded));
  FOLD_BTN.title = folded ? 'Show the navigation (\\)' : 'Hide the navigation (\\)';
  try { localStorage.setItem(FOLD_KEY, folded ? '1' : '0'); } catch (e) { /* private mode */ }
}

function initFold() {
  let folded = false;
  try { folded = localStorage.getItem(FOLD_KEY) === '1'; } catch (e) { /* ignore */ }
  applyFold(folded);
  FOLD_BTN.addEventListener('click', () => applyFold(!SHELL.classList.contains('folded')));
  document.addEventListener('keydown', (e) => {
    const typing = /^(INPUT|TEXTAREA)$/.test(e.target.tagName) || e.target.isContentEditable;
    if (e.key === '\\' && !typing && !e.metaKey && !e.ctrlKey) {
      e.preventDefault();
      applyFold(!SHELL.classList.contains('folded'));
    }
  });
}

/* --------------------------------------------------------------------- nav */

function buildNav() {
  NAV.replaceChildren();
  for (const t of topics) {
    const isOpen = t.id === openTopic;

    const btn = el('button', {
      class: 'topic-btn',
      type: 'button',
      id: 'topic-' + t.id,
      'aria-expanded': String(isOpen),
      'aria-controls': 'papers-' + t.id,
    }, [
      el('span', { text: t.title }),
      el('span', { class: 'count', text: '(' + t.papers.length + ')' }),
    ]);
    btn.addEventListener('click', () => {
      openTopic = isOpen ? null : t.id;
      buildNav();
      if (!isOpen) showTopic(t.id);
    });

    const list = el('div', {
      class: 'paper-list',
      id: 'papers-' + t.id,
      role: 'tablist',
      'aria-label': t.title + ' papers',
    });
    if (!isOpen) list.hidden = true;

    for (const p of ordered(t.papers)) {
      const tab = el('button', {
        class: 'tab',
        type: 'button',
        role: 'tab',
        id: 'tab-' + p.file,
        'aria-selected': String(current && current.paper.file === p.file),
      }, [
        el('span', { class: 'year', text: String(p.year) }),
        el('span', { class: 'who', text: firstAuthor(p.authors) }),
        el('span', { class: 'what', text: p.one_line || '' }),
      ]);
      tab.addEventListener('click', () => showPaper(t, p));
      tab.addEventListener('keydown', (e) => railKeys(e, list));
      list.appendChild(tab);
    }

    NAV.appendChild(el('div', { class: 'topic-group' }, [btn, list]));
  }
}

/* ArrowUp/ArrowDown move between papers with wraparound, as in v1. */
function railKeys(e, list) {
  if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
  e.preventDefault();
  const tabs = Array.from(list.querySelectorAll('.tab'));
  const i = tabs.indexOf(e.currentTarget);
  const next = e.key === 'ArrowDown'
    ? tabs[(i + 1) % tabs.length]
    : tabs[(i - 1 + tabs.length) % tabs.length];
  next.focus();
  next.click();
}

function markSelected(file) {
  for (const t of NAV.querySelectorAll('.tab')) {
    t.setAttribute('aria-selected', String(t.id === 'tab-' + file));
  }
}

/* ------------------------------------------------------------------- views */

function showTopic(id) {
  const t = topics.find((x) => x.id === id);
  if (!t) return;
  current = null;
  markSelected(null);

  const rows = ordered(t.papers).map((p) => {
    const link = el('a', { href: '#', text: p.title });
    link.addEventListener('click', (e) => { e.preventDefault(); showPaper(t, p); });
    return el('tr', {}, [
      el('td', { class: 'yr', text: String(p.year) }),
      el('td', { text: firstAuthor(p.authors) + (String(p.authors).includes(',') ? ' et al.' : '') }),
      el('td', {}, [link, el('div', { class: 'what', text: p.one_line || '' })]),
      el('td', { text: p.venue || '' }),
    ]);
  });

  VIEW.replaceChildren(
    el('h1', { text: t.title }),
    el('p', { class: 'standfirst', text: t.overview || '' }),
    el('div', { class: 'table-wrap' }, [
      el('table', { class: 'papers' }, [
        el('thead', {}, [el('tr', {}, [
          el('th', { text: 'Year' }), el('th', { text: 'Authors' }),
          el('th', { text: 'Paper' }), el('th', { text: 'Venue' }),
        ])]),
        el('tbody', {}, rows),
      ]),
    ])
  );
  document.title = t.title + ' — Paper summaries';
}

async function showPaper(topic, paper) {
  current = { topic, paper };
  markSelected(paper.file);
  const url = [PAPERS_ROOT, topic.id, paper.file].join('/');

  let md;
  try {
    const r = await fetch(url);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    md = await r.text();
  } catch (err) {
    msg('No summary file yet for this paper. Expected it at', url);
    return;
  }

  const doc = el('div', { class: 'doc' });
  doc.innerHTML = marked.parse(md, { gfm: true, breaks: false });

  // Mark the one section that may hold the reader's own words, and give every
  // heading an id so a comment can name where it was made.
  for (const h of doc.querySelectorAll('h2')) {
    if (/relevance to this project/i.test(h.textContent)) h.classList.add('own-reading');
  }

  const cite = el('p', {
    class: 'cite',
    text: [paper.authors, '(' + paper.year + ').', paper.title + '.', paper.venue + '.',
      paper.doi ? 'doi:' + paper.doi : '', '· PDF: ' + paper.pdf].filter(Boolean).join(' '),
  });

  VIEW.replaceChildren(doc, cite);
  buildDock();

  try {
    renderMathInElement(doc, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\[', right: '\\]', display: true },
        { left: '\\(', right: '\\)', display: false },
      ],
      throwOnError: false,
    });
  } catch (err) {
    /* Maths failing to render must not blank the prose. */
    console.warn('KaTeX did not run:', err);
  }

  document.title = firstAuthor(paper.authors) + ' ' + paper.year + ' — Paper summaries';
  window.scrollTo({ top: 0 });
  loadComments();
}

/* ---------------------------------------------------------------- comments */

/* Which `## ` section a DOM node sits under, so a note records its place. */
function sectionOf(node) {
  let n = node && node.nodeType === 3 ? node.parentNode : node;
  while (n && n !== VIEW) {
    for (let s = n.previousElementSibling; s; s = s.previousElementSibling) {
      if (s.tagName === 'H2') return s.textContent.trim();
    }
    n = n.parentNode;
  }
  return '';
}

function buildDock() {
  if (document.getElementById('dock')) return;

  const badge = el('span', { class: 'badge', id: 'dockcount', text: '0' });
  const handle = el('button', {
    class: 'dock-handle', type: 'button', id: 'docktoggle',
    'aria-expanded': 'false', 'aria-controls': 'dockbody',
  }, [el('span', { text: 'Notes' }), badge]);

  const quoted = el('div', { id: 'cquote', class: 'cquote' });
  quoted.hidden = true;
  const box = el('textarea', {
    id: 'cbox', rows: '3',
    placeholder: 'Select text in the summary to quote it, or just write a note\u2026',
    'aria-label': 'New note',
  });
  const save = el('button', { class: 'btn', type: 'button', text: 'Save note' });
  const clear = el('button', { class: 'btn ghost', type: 'button', text: 'Clear quote' });
  save.addEventListener('click', addComment);
  clear.addEventListener('click', () => { pendingQuote = ''; pendingSection = ''; renderQuote(); });

  const body = el('div', { class: 'dock-body', id: 'dockbody' }, [
    el('div', { id: 'clist', class: 'dock-list' }),
    el('div', { class: 'cform' }, [quoted, box, el('div', { class: 'crow' }, [save, clear])]),
  ]);
  body.hidden = true;

  const dock = el('aside', { class: 'dock', id: 'dock', 'aria-label': 'Notes' }, [handle, body]);
  document.body.appendChild(dock);

  handle.addEventListener('click', () => setDock(body.hidden));
}

function setDock(open) {
  const body = document.getElementById('dockbody');
  const handle = document.getElementById('docktoggle');
  if (!body) return;
  body.hidden = !open;
  handle.setAttribute('aria-expanded', String(open));
  document.getElementById('dock').classList.toggle('open', open);
  if (open) { const b = document.getElementById('cbox'); if (b) b.focus(); }
}

let pendingQuote = '';
let pendingSection = '';

function renderQuote() {
  const q = document.getElementById('cquote');
  if (!q) return;
  q.hidden = !pendingQuote;
  q.textContent = pendingQuote ? '“' + pendingQuote + '”' : '';
}

/* Capture a selection inside the rendered summary as the quote for the next note. */
document.addEventListener('mouseup', () => {
  const sel = window.getSelection();
  if (!sel || sel.isCollapsed) return;
  const doc = VIEW.querySelector('.doc');
  if (!doc || !doc.contains(sel.anchorNode)) return;
  const text = sel.toString().trim().replace(/\s+/g, ' ');
  if (!text) return;
  pendingQuote = text.slice(0, 2000);
  pendingSection = sectionOf(sel.anchorNode);
  renderQuote();
  setDock(true);
});

async function loadComments() {
  if (!current) return;
  try {
    const r = await fetch('/api/comments?paper=' + encodeURIComponent(current.paper.file));
    comments = r.ok ? (await r.json()).comments || [] : [];
  } catch (err) {
    comments = [];
    renderComments(true);
    return;
  }
  renderComments(false);
}

function renderComments(offline) {
  const list = document.getElementById('clist');
  const count = document.getElementById('dockcount');
  if (!list) return;

  if (offline) {
    list.replaceChildren(el('p', { class: 'msg' }, [
      el('span', { text: 'Notes need the app server. Start it with ' }),
      el('code', { text: './run_summary_app.sh' }),
    ]));
    const form = document.querySelector('.cform');
    if (form) form.hidden = true;
    if (count) count.textContent = '-';
    return;
  }

  const open = comments.filter((c) => c.status !== 'resolved').length;
  if (count) {
    count.textContent = String(open);
    count.classList.toggle('zero', open === 0);
  }

  if (!comments.length) {
    list.replaceChildren(el('p', { class: 'cempty', text: 'No notes on this paper yet.' }));
  } else {
    list.replaceChildren(...comments.map((c, i) => {
      const done = c.status === 'resolved';
      const toggle = el('button', { class: 'btn tiny', type: 'button', text: done ? 'Reopen' : 'Resolve' });
      toggle.addEventListener('click', () => setStatus(c.id, done ? 'open' : 'resolved'));
      const del = el('button', { class: 'btn tiny ghost', type: 'button', text: 'Delete' });
      del.addEventListener('click', () => removeComment(c.id));
      const jump = el('button', { class: 'btn tiny ghost', type: 'button', text: 'Show' });
      jump.addEventListener('click', () => {
        const pin = VIEW.querySelector('.note-pin[data-id="' + c.id + '"]');
        if (pin) { pin.scrollIntoView({ block: 'center', behavior: 'smooth' }); pin.click(); }
      });
      return el('article', { class: 'comment' + (done ? ' resolved' : '') }, [
        el('div', { class: 'cmeta', text: (i + 1) + ' · ' + (c.section || 'general') }),
        c.quote ? el('blockquote', { class: 'cquote', text: '\u201c' + c.quote + '\u201d' }) : null,
        el('p', { class: 'ctext', text: c.text }),
        el('div', { class: 'crow' }, [c.quote ? jump : null, toggle, del].filter(Boolean)),
      ]);
    }));
  }
  anchorNotes();
}

/* Pin each quoted note next to the text it refers to. Runs after KaTeX so the
   offsets match what the reader selected. */
function anchorNotes() {
  const doc = VIEW.querySelector('.doc');
  if (!doc || !window.Notes) return;
  Notes.clearPins(doc);
  Notes.closePopover();
  comments.forEach((c, i) => {
    if (!c.quote) return;
    const range = Notes.rangeForQuote(doc, c.quote);
    if (!range) return;
    const pin = Notes.pinRange(range, i + 1, (p) => Notes.openPopover(p, c, {
      setStatus, remove: removeComment,
    }));
    if (pin) {
      pin.dataset.id = c.id;
      if (c.status === 'resolved') pin.classList.add('resolved');
    }
  });
}

async function post(body) {
  const r = await fetch('/api/comments', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || 'HTTP ' + r.status);
  return r.json();
}

async function addComment() {
  const box = document.getElementById('cbox');
  const text = (box.value || '').trim();
  if (!text) { box.focus(); return; }
  try {
    await post({
      action: 'add', paper: current.paper.file, topic: current.topic.id,
      section: pendingSection, quote: pendingQuote, text,
    });
    box.value = '';
    pendingQuote = ''; pendingSection = '';
    renderQuote();
    await loadComments();
    setDock(true);
  } catch (err) {
    alert('Could not save the note: ' + err.message);
  }
}

async function setStatus(id, status) {
  try {
    const store = await post({ action: 'status', paper: current.paper.file, id, status });
    comments = store.comments || [];
    renderComments(false);
  } catch (err) { alert('Could not update: ' + err.message); }
}

async function removeComment(id) {
  try {
    const store = await post({ action: 'delete', paper: current.paper.file, id });
    comments = store.comments || [];
    renderComments(false);
  } catch (err) { alert('Could not delete: ' + err.message); }
}

/* -------------------------------------------------------------------- boot */

(async function init() {
  initFold();

  let data;
  try {
    const r = await fetch(DATA_URL);
    if (!r.ok) throw new Error('HTTP ' + r.status);
    data = await r.json();
  } catch (err) {
    msg('Could not load papers/topics.json. Serve the repository over HTTP — '
      + 'opening this file directly will not work. Run', './run_summary_app.sh');
    return;
  }

  topics = (data.topics || []).filter((t) => t && t.id);
  if (!topics.length) { msg('papers/topics.json lists no topics.'); return; }

  openTopic = topics[0].id;
  buildNav();
  showTopic(openTopic);
})();
