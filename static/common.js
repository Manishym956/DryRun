// Shared helpers. Always build DOM with textContent: model output is untrusted.
const $ = id => document.getElementById(id);

function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text != null) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}

async function api(path, opts) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res.headers.get('content-type')?.includes('json') ? res.json() : res;
}

const fmtTime = s => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
const fmtDate = ts => new Date(ts * 1000).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
const PERSONA_LABELS = { friendly: 'Friendly mentor', bar_raiser: 'Tough bar-raiser', startup_cto: 'Startup CTO' };
