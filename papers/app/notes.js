/* Notes: a floating dock, plus anchoring each note beside the text it quotes.
 *
 * Kept separate from app.js because it is the only part that reaches into the
 * rendered markdown and rewrites it. Everything here degrades: if a quote can
 * no longer be found in the text, the note still shows in the dock, just
 * without a pin.
 */

/* ------------------------------------------------- finding a quote in the DOM */

/* Build a whitespace-normalised copy of the text under `root`, remembering
 * where each character came from, so a quote captured from a selection can be
 * mapped back to a Range. KaTeX subtrees are skipped: they carry duplicate
 * text in MathML that would match spuriously. */
function textIndex(root) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(n) {
      if (!n.nodeValue) return NodeFilter.FILTER_REJECT;
      if (n.parentElement && n.parentElement.closest('.katex, .note-pin')) {
        return NodeFilter.FILTER_REJECT;
      }
      return NodeFilter.FILTER_ACCEPT;
    },
  });

  let text = '';
  const map = [];            // map[i] = [node, offsetInNode] for text[i]
  let lastWasSpace = true;   // collapse leading whitespace too
  let n;
  while ((n = walker.nextNode())) {
    const v = n.nodeValue;
    for (let i = 0; i < v.length; i++) {
      const isSpace = /\s/.test(v[i]);
      if (isSpace) {
        if (lastWasSpace) continue;
        text += ' ';
        map.push([n, i]);
        lastWasSpace = true;
      } else {
        text += v[i];
        map.push([n, i]);
        lastWasSpace = false;
      }
    }
  }
  return { text, map };
}

function rangeForQuote(root, quote) {
  const needle = String(quote || '').trim().replace(/\s+/g, ' ');
  if (needle.length < 4) return null;
  const { text, map } = textIndex(root);
  const at = text.indexOf(needle);
  if (at < 0) return null;

  const [startNode, startOff] = map[at];
  const [endNode, endOff] = map[at + needle.length - 1];
  const r = document.createRange();
  try {
    r.setStart(startNode, startOff);
    r.setEnd(endNode, endOff + 1);
  } catch (err) {
    return null;
  }
  return r;
}

/* ------------------------------------------------------------------ pinning */

/* Wrap a range in <mark> and drop a numbered pin after it. Returns the pin, or
 * null when the range crosses element boundaries in a way surroundContents
 * cannot handle — in which case the note simply has no pin. */
function pinRange(range, n, onClick) {
  const mark = document.createElement('mark');
  mark.className = 'anno';
  try {
    range.surroundContents(mark);
  } catch (err) {
    return null;
  }
  const pin = document.createElement('button');
  pin.type = 'button';
  pin.className = 'note-pin';
  pin.textContent = String(n);
  pin.title = 'Note ' + n;
  pin.setAttribute('aria-label', 'Note ' + n);
  pin.addEventListener('click', (e) => { e.stopPropagation(); onClick(pin); });
  mark.after(pin);
  return pin;
}

function clearPins(root) {
  for (const p of root.querySelectorAll('.note-pin')) p.remove();
  for (const m of root.querySelectorAll('mark.anno')) {
    const parent = m.parentNode;
    while (m.firstChild) parent.insertBefore(m.firstChild, m);
    m.remove();
    parent.normalize();
  }
}

/* --------------------------------------------------------------- a popover */

let popover = null;

function closePopover() {
  if (popover) { popover.remove(); popover = null; }
}

/* Place the note beside its pin: to the right when the margin allows it,
 * otherwise directly above. */
function openPopover(pin, note, actions) {
  closePopover();

  const box = document.createElement('div');
  box.className = 'note-pop';
  box.setAttribute('role', 'dialog');
  box.setAttribute('aria-label', 'Note');

  const meta = document.createElement('div');
  meta.className = 'cmeta';
  meta.textContent = [note.section || 'general',
    note.created.replace('T', ' ').replace('+00:00', ' UTC')].join(' · ');

  const body = document.createElement('p');
  body.className = 'ctext';
  body.textContent = note.text;

  const row = document.createElement('div');
  row.className = 'crow';
  const done = note.status === 'resolved';
  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'btn tiny';
  toggle.textContent = done ? 'Reopen' : 'Resolve';
  toggle.addEventListener('click', () => actions.setStatus(note.id, done ? 'open' : 'resolved'));
  const del = document.createElement('button');
  del.type = 'button';
  del.className = 'btn tiny ghost';
  del.textContent = 'Delete';
  del.addEventListener('click', () => actions.remove(note.id));
  const shut = document.createElement('button');
  shut.type = 'button';
  shut.className = 'btn tiny ghost';
  shut.textContent = 'Close';
  shut.addEventListener('click', closePopover);
  row.append(toggle, del, shut);

  box.append(meta, body, row);
  document.body.appendChild(box);

  const r = pin.getBoundingClientRect();
  const w = box.offsetWidth;
  const h = box.offsetHeight;
  const gap = 12;
  const room = window.innerWidth - (r.right + gap);

  let left;
  let top;
  if (room >= w + 16) {                       // to the right, preferred
    left = r.right + gap;
    top = r.top + window.scrollY - 8;
    box.classList.add('at-right');
  } else {                                    // otherwise above
    left = Math.max(12, Math.min(r.left - w / 2, window.innerWidth - w - 12));
    top = r.top + window.scrollY - h - gap;
    box.classList.add('at-top');
  }
  box.style.left = left + 'px';
  box.style.top = Math.max(window.scrollY + 8, top) + 'px';

  popover = box;
  requestAnimationFrame(() => box.classList.add('in'));
}

document.addEventListener('click', (e) => {
  if (popover && !popover.contains(e.target) && !e.target.closest('.note-pin')) closePopover();
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closePopover(); });
window.addEventListener('resize', closePopover);

window.Notes = { rangeForQuote, pinRange, clearPins, openPopover, closePopover };
