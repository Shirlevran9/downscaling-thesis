/* Paper summaries — reading app.
 *
 * Reads papers/topics.json, builds a two-level left nav (topic -> papers), and
 * renders a paper's markdown on demand. No build step and no framework: the
 * summaries are the artefact, this only displays them.
 *
 * The roving-focus keyboard handling follows history/summaries_html_v1.html.
 */

const NAV = document.getElementById('nav');
const VIEW = document.getElementById('view');
const DATA_URL = '../topics.json';   // relative to papers/app/
const PAPERS_ROOT = '..';            // papers/

let topics = [];
let openTopic = null;

/* ------------------------------------------------------------------ helpers */

function el(tag, props = {}, children = []) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') n.className = v;
    else if (k === 'text') n.textContent = v;
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

/* Sort by year, then by first author — the order stated in the papers rule. */
function ordered(papers) {
  return papers.slice().sort((a, b) => a.year - b.year || a.authors.localeCompare(b.authors));
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
        'aria-selected': 'false',
      }, [
        el('span', { class: 'year', text: String(p.year) }),
        el('span', { class: 'who', text: p.authors.split(',')[0] }),
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
  markSelected(null);

  const rows = ordered(t.papers).map((p) => {
    const link = el('a', { href: '#', text: p.title });
    link.addEventListener('click', (e) => { e.preventDefault(); showPaper(t, p); });
    return el('tr', {}, [
      el('td', { class: 'yr', text: String(p.year) }),
      el('td', { text: p.authors.split(',')[0] + (p.authors.includes(',') ? ' et al.' : '') }),
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

  // Mark the one section that may hold the reader's own words.
  for (const h of doc.querySelectorAll('h2')) {
    if (/relevance to this project/i.test(h.textContent)) h.classList.add('own-reading');
  }

  VIEW.replaceChildren(
    doc,
    el('p', { class: 'cite', text: [paper.authors, '(' + paper.year + ').', paper.title + '.',
      paper.venue + '.', paper.doi ? 'doi:' + paper.doi : '', '· PDF: ' + paper.pdf]
      .filter(Boolean).join(' ') })
  );

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

  document.title = paper.authors.split(',')[0] + ' ' + paper.year + ' — Paper summaries';
  VIEW.parentElement.scrollTo({ top: 0 });
}

/* -------------------------------------------------------------------- boot */

(async function init() {
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
