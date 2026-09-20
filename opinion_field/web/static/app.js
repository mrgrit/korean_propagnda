/* K-Propaganda web UI — vanilla JS, no build step. Charts are hand-rolled SVG (line, heatmap, bars). */
'use strict';
const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const fmtP = (v, d = 4) => (v == null ? '–' : (+v).toFixed(d));
const fmtPct = (v) => (v == null ? '–' : (v * 100).toFixed(2) + '%');
const fmtN = (v) => (v == null ? '–' : Math.round(v).toLocaleString('ko-KR'));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const STATE_KO = { abstain: '기권', B_strong: 'B 강', B_weak: 'B 약', undecided: '부동', A_weak: 'A 약', A_strong: 'A 강' };
const CH_KO = { youtube: '유튜브', portal: '포털', kakao: '카톡', instagram: '인스타', tv: 'TV', wom: '입소문' };

const state = { status: null, config: null, run: null, runName: null, round: 1, jobId: null, jobLines: 0, playing: null, lastRunReload: 0, fileKind: 'campaign' };

async function api(url, opts = {}) {
  const r = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...opts });
  if (r.status === 401) { location.href = '/login'; throw new Error('unauthorized'); }
  const ct = r.headers.get('content-type') || '';
  if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (_) { } throw new Error(m); }
  return ct.includes('json') ? r.json() : r.text();
}
function toast(msg, isErr = false) { const h = $('#run-hint'); h.textContent = msg; h.style.color = isErr ? css('--critical') : ''; }

/* ───────────────────────── status / runs ───────────────────────── */
async function loadStatus() {
  const st = await api('/api/status');
  state.status = st;
  const chips = [];
  chips.push(`<span class="chip ${st.data.length ? 'ok' : 'bad'}"><i class="dot"></i>데이터 ${st.data.length}개</span>`);
  chips.push(`<span class="chip ${st.claude.available && st.claude.credentials ? 'ok' : 'bad'}"><i class="dot"></i>Claude Code ${st.claude.available ? (st.claude.version || '').split(' ')[0] : '없음'}${st.claude.available && !st.claude.credentials ? ' (로그인 필요)' : ''}</span>`);
  chips.push(`<span class="chip ${st.busy ? 'busy' : ''}"><i class="dot"></i>${st.job_current ? `작업 중: ${st.job_current.kind} ${st.job_current.params.name || ''}` : '대기'}</span>`);
  $('#status-chips').innerHTML = chips.join('');
  $('#data-list').innerHTML = st.data.length ? st.data.map((d) => `<div class="row"><span>${d.dir}</span><span class="muted">${fmtN(d.n_voters)}명 · ${d.source && d.source.startsWith('SYNTHETIC') ? '합성 인구' : 'Nemotron'} · 지역표심 ${d.priors && String(d.priors.regional).includes('PLACEHOLDER') ? '자리표시자' : '실측'}</span></div>`).join('')
    : `<div class="muted">빌드된 데이터가 없습니다. ${st.raw_shards ? `Nemotron 샤드 ${st.raw_shards}개 있음 → 그냥 빌드` : '샤드 없음 → 합성 인구 N을 넣고 빌드'}</div>`;
  const sel = $('#r-data'); const prev = sel.value;
  sel.innerHTML = st.data.map((d) => `<option value="${d.dir}">${d.dir} (${fmtN(d.n_voters)})</option>`).join('');
  if (prev && st.data.some((d) => d.dir === prev)) sel.value = prev; else if (st.data.some((d) => d.dir === 'data/processed')) sel.value = 'data/processed';
  renderRunSelect(st.runs);
  if (st.job_current && state.jobId !== st.job_current.id) attachJob(st.job_current.id);
  $('#btn-stop').hidden = !st.job_current;
}
function renderRunSelect(runs) {
  const sel = $('#run-select'); const prev = state.runName;
  sel.innerHTML = runs.length ? runs.map((r) => {
    const s = r.summary || {}; const tag = r.kind === 'compare' ? '대조' : (s.finished ? '완료' : (s.stopped ? '중단' : `${s.rounds_done}/${s.rounds_total}`));
    return `<option value="${r.name}">${r.name} · ${tag} · ${s.backend || ''} · A ${fmtP(s.A, 3)}</option>`;
  }).join('') : '<option value="">(실행 없음)</option>';
  if (prev && runs.some((r) => r.name === prev)) sel.value = prev;
  else if (runs.length) { sel.value = runs[0].name; loadRun(runs[0].name); }
}
async function loadRun(name) {
  if (!name) return;
  state.runName = name;
  try { state.run = await api(`/api/runs/${encodeURIComponent(name)}`); } catch (e) { toast(e.message, true); return; }
  const R = state.run.rounds.length;
  const sl = $('#hm-round'); sl.max = Math.max(R, 1); if (state.round > R || state.round < 1 || !state._roundPinned) state.round = R;
  sl.value = state.round;
  $('#center-run').textContent = `— ${name}${state.run.sub ? ' / ' + state.run.sub : ''} · ${fmtN(state.run.manifest.data && state.run.manifest.data.n_voters)}명`;
  renderAll();
}
function renderAll() {
  if (!state.run || !state.run.rounds.length) { $('#kpis').innerHTML = '<div class="muted">라운드 기록이 아직 없습니다.</div>'; return; }
  renderKPIs(); renderTrajectory(); renderCompare(); renderStateChart(); renderHeatmap(); renderChannelBars(); renderPlan(); renderContrib(); renderBattleLog();
  $('#hm-round-label').textContent = `R${state.round} / ${state.run.rounds.length}`;
}
const rec = () => state.run.rounds[state.round - 1];

/* ───────────────────────── KPI tiles ───────────────────────── */
function renderKPIs() {
  const r = rec(); const p = state.run.rounds[state.round - 2];
  const d = (a, b, fmt = fmtPct) => (b == null ? '' : `<div class="d ${a - b > 0 ? 'up' : a - b < 0 ? 'down' : ''}">${a - b >= 0 ? '+' : ''}${fmt(a - b)} vs 이전</div>`);
  const tiles = [
    ['후보 A 지지율', fmtPct(r.support.A), d(r.support.A, p && p.support.A)],
    ['후보 B 지지율', fmtPct(r.support.B), d(r.support.B, p && p.support.B)],
    ['마진 (A−B)', fmtPct(r.support.A - r.support.B), `<div class="d">목표 ≥ ${fmtPct(state.run.manifest.config.goal_margin)} · ${r.goal_round ? `R${r.goal_round} 달성` : '미달'}</div>`],
    ['부동층 (전체 대비)', fmtPct(r.support.undecided_all), d(r.support.undecided_all, p && p.support.undecided_all)],
    ['노출 (이번 라운드)', fmtN(r.exposure.n_exposed), `<div class="d">후보 셀 ${fmtN(r.exposure.n_candidates)}</div>`],
    ['L2 승격 / 폴백', `${fmtN(r.l2.n_promoted + (r.l2.n_l3_promoted || 0))}`, `<div class="d">${r.l2.n_fallback ? `<span style="color:var(--critical)">폴백 ${r.l2.n_fallback}</span>` : `↑${r.l2.n_delta_up} ↓${r.l2.n_delta_down} 공유 ${r.l2.n_share}`}</div>`],
    ['누적 비용', fmtN(r.cost.cumulative), `<div class="d">방어 ${fmtN(r.cost.defender_cumulative)}</div>`],
    ['방어 되돌림', fmtN(r.defense.n_reverted), `<div class="d">${r.defense.detected ? `탐지 p=${fmtP(r.defense.p_detect, 2)} · 정정노출 ${fmtN(r.defense.n_corr_exposed)}` : `미탐지 p=${fmtP(r.defense.p_detect, 2)}`}</div>`],
  ];
  $('#kpis').innerHTML = tiles.map(([k, v, dd]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div>${dd || ''}</div>`).join('');
}

/* ───────────────────────── SVG helpers ───────────────────────── */
const tip = $('#tooltip');
function showTip(html, x, y) { tip.innerHTML = html; tip.hidden = false; const w = tip.offsetWidth, h = tip.offsetHeight; tip.style.left = Math.min(x + 12, innerWidth - w - 8) + 'px'; tip.style.top = Math.min(y + 12, innerHeight - h - 8) + 'px'; }
function hideTip() { tip.hidden = true; }
function svgEl(w, h) { const s = document.createElementNS('http://www.w3.org/2000/svg', 'svg'); s.setAttribute('viewBox', `0 0 ${w} ${h}`); s.setAttribute('width', w); s.setAttribute('height', h); s.style.height = h + 'px'; return s; }
function el(tag, attrs = {}, text) { const e = document.createElementNS('http://www.w3.org/2000/svg', tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (text != null) e.textContent = text; return e; }
function niceTicks(min, max, n = 4) { const span = max - min || 1; const step0 = span / n; const mag = Math.pow(10, Math.floor(Math.log10(step0))); const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || mag * 10; const lo = Math.floor(min / step) * step; const out = []; for (let v = lo; v <= max + 1e-9; v += step) out.push(+v.toFixed(10)); return out; }

/** Line chart with legend, direct end labels, crosshair + tooltip. series: [{name, color, values}] */
function lineChart(container, { x, series, fmt = (v) => fmtPct(v), height = 230, yMin, yMax, highlightX, refY, refLabel }) {
  container.innerHTML = '';
  const W = Math.max(container.clientWidth || 600, 280), H = height, m = { l: 52, r: 100, t: 12, b: 28 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const all = series.flatMap((s) => s.values).filter((v) => v != null); if (refY != null) all.push(refY);
  let lo = yMin != null ? yMin : Math.min(...all), hi = yMax != null ? yMax : Math.max(...all);
  if (hi - lo < 1e-9) { lo -= 0.01; hi += 0.01; } const pad = (hi - lo) * 0.08; lo -= pad; hi += pad;
  const xs = (i) => m.l + (x.length > 1 ? (i / (x.length - 1)) * iw : iw / 2);
  const ys = (v) => m.t + ih - ((v - lo) / (hi - lo)) * ih;
  const svg = svgEl(W, H);
  niceTicks(lo, hi, 4).forEach((t) => { if (t < lo || t > hi) return; svg.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: ys(t), y2: ys(t), stroke: css('--grid'), 'stroke-width': 1 })); svg.appendChild(el('text', { x: m.l - 6, y: ys(t) + 4, 'text-anchor': 'end', 'font-size': 11, fill: css('--muted') }, fmt(t))); });
  svg.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, stroke: css('--axis') }));
  const step = Math.max(1, Math.ceil(x.length / Math.max(3, Math.floor(iw / 46)))); const lastI = x.length - 1;
  x.forEach((xv, i) => { const regular = i % step === 0 && (lastI - i >= step / 2 || i === lastI); if (regular || i === lastI) svg.appendChild(el('text', { x: xs(i), y: H - 8, 'text-anchor': 'middle', 'font-size': 11, fill: css('--muted') }, 'R' + xv)); });
  if (refY != null) { svg.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: ys(refY), y2: ys(refY), stroke: css('--text-2'), 'stroke-width': 1, 'stroke-dasharray': '4 3' })); svg.appendChild(el('text', { x: W - m.r - 4, y: ys(refY) - 4, 'text-anchor': 'end', 'font-size': 10, fill: css('--text-2') }, refLabel || fmt(refY))); }
  const endLabels = [];
  series.forEach((s) => {
    const d = s.values.map((v, i) => (v == null ? null : `${i === 0 || s.values[i - 1] == null ? 'M' : 'L'}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`)).filter(Boolean).join(' ');
    svg.appendChild(el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', 'stroke-dasharray': s.dash || 'none' }));
    const last = s.values.length - 1;
    if (s.values[last] != null && series.length <= 4) endLabels.push({ y: ys(s.values[last]), text: `${s.name} ${fmt(s.values[last])}`, x: xs(last) + 6 });
  });
  endLabels.sort((a, b) => a.y - b.y).forEach((lb, i, arr) => { if (i > 0 && lb.y - arr[i - 1].y < 13) lb.y = arr[i - 1].y + 13; });
  endLabels.forEach((lb) => svg.appendChild(el('text', { x: lb.x, y: lb.y + 4, 'font-size': 11, fill: css('--text-2') }, lb.text)));
  if (highlightX != null) { const i = x.indexOf(highlightX); if (i >= 0) svg.appendChild(el('line', { x1: xs(i), x2: xs(i), y1: m.t, y2: m.t + ih, stroke: css('--axis'), 'stroke-dasharray': '3 3' })); }
  const cross = el('g', { visibility: 'hidden' }); const vline = el('line', { y1: m.t, y2: m.t + ih, stroke: css('--text-2'), 'stroke-width': 1 }); cross.appendChild(vline);
  const dots = series.map((s) => { const c = el('circle', { r: 4.5, fill: s.color, stroke: css('--surface-1'), 'stroke-width': 2 }); cross.appendChild(c); return c; });
  svg.appendChild(cross);
  const hit = el('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent' }); svg.appendChild(hit);
  hit.addEventListener('mousemove', (ev) => { const bb = svg.getBoundingClientRect(); const px = ((ev.clientX - bb.left) / bb.width) * W; const i = Math.max(0, Math.min(x.length - 1, Math.round(((px - m.l) / iw) * (x.length - 1)))); cross.setAttribute('visibility', 'visible'); vline.setAttribute('x1', xs(i)); vline.setAttribute('x2', xs(i)); dots.forEach((d, k) => { const v = series[k].values[i]; if (v == null) d.setAttribute('visibility', 'hidden'); else { d.setAttribute('visibility', 'visible'); d.setAttribute('cx', xs(i)); d.setAttribute('cy', ys(v)); } }); showTip(`<div class="t">라운드 ${x[i]}</div>` + series.map((s, k) => `<div><span class="sw" style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${s.color};margin-right:6px"></span>${s.name}: <b>${fmt(s.values[i])}</b></div>`).join(''), ev.clientX, ev.clientY); });
  hit.addEventListener('mouseleave', () => { cross.setAttribute('visibility', 'hidden'); hideTip(); });
  hit.addEventListener('click', (ev) => { const bb = svg.getBoundingClientRect(); const px = ((ev.clientX - bb.left) / bb.width) * W; const i = Math.max(0, Math.min(x.length - 1, Math.round(((px - m.l) / iw) * (x.length - 1)))); setRound(x[i]); });
  if (series.length >= 2) { const lg = document.createElement('div'); lg.className = 'legend'; lg.innerHTML = series.map((s) => `<span><i class="sw" style="background:${s.color}"></i>${s.name}</span>`).join(''); container.appendChild(lg); }
  container.appendChild(svg);
}

function mixHex(a, b, t) { const pa = [1, 3, 5].map((i) => parseInt(a.slice(i, i + 2), 16)); const pb = [1, 3, 5].map((i) => parseInt(b.slice(i, i + 2), 16)); return '#' + pa.map((v, i) => Math.round(v + (pb[i] - v) * t).toString(16).padStart(2, '0')).join(''); }
function divergingColor(v, vmax) { const mid = css('--div-mid'), pos = css('--div-pos'), neg = css('--div-neg'); if (!vmax) return mid; const t = Math.max(-1, Math.min(1, v / vmax)); return t >= 0 ? mixHex(mid, pos, Math.sqrt(t)) : mixHex(mid, neg, Math.sqrt(-t)); }
function sequentialColor(v, vmax) { const lo = css('--seq-100'), hi = css('--seq-700'); if (!vmax) return lo; return mixHex(lo, hi, Math.sqrt(Math.max(0, Math.min(1, v / vmax)))); }

/** Heatmap rows × cols with hover tooltip; diverging or sequential */
function heatmap(container, { rows, cols, values, counts, mode, fmt, title }) {
  container.innerHTML = '';
  const W = Math.max(container.clientWidth || 600, 320), labelW = 56, top = 24, cellH = 20, gap = 2;
  const cellW = (W - labelW - 8) / cols.length;
  const H = top + rows.length * cellH + 44;
  const svg = svgEl(W, H);
  const flat = values.flat().filter((v) => counts ? true : v != null);
  const vmax = Math.max(1e-9, ...flat.map((v) => Math.abs(v)));
  cols.forEach((c, j) => svg.appendChild(el('text', { x: labelW + j * cellW + cellW / 2, y: 14, 'text-anchor': 'middle', 'font-size': 11, fill: css('--muted') }, c)));
  rows.forEach((r, i) => {
    svg.appendChild(el('text', { x: labelW - 6, y: top + i * cellH + cellH / 2 + 4, 'text-anchor': 'end', 'font-size': 11, fill: css('--text-2') }, r));
    cols.forEach((c, j) => {
      const v = values[i][j]; const n = counts ? counts[i][j] : null;
      const color = n === 0 ? css('--page') : (mode === 'diverging' ? divergingColor(v, vmax) : sequentialColor(v, vmax));
      const rect = el('rect', { x: labelW + j * cellW + gap / 2, y: top + i * cellH + gap / 2, width: cellW - gap, height: cellH - gap, rx: 3, fill: color });
      rect.addEventListener('mousemove', (ev) => showTip(`<div class="t">${r} · ${c}</div><div>${title}: <b>${fmt(v)}</b></div>${n != null ? `<div class="t">인구 ${fmtN(n)}</div>` : ''}`, ev.clientX, ev.clientY));
      rect.addEventListener('mouseleave', hideTip);
      svg.appendChild(rect);
    });
  });
  // legend bar
  const ly = H - 30, lx = labelW, lw = Math.min(220, W - labelW - 8);
  const steps = 24;
  for (let k = 0; k < steps; k++) { const t = k / (steps - 1); const v = mode === 'diverging' ? (t * 2 - 1) * vmax : t * vmax; svg.appendChild(el('rect', { x: lx + (k / steps) * lw, y: ly, width: lw / steps + 0.5, height: 8, fill: mode === 'diverging' ? divergingColor(v, vmax) : sequentialColor(v, vmax) })); }
  svg.appendChild(el('text', { x: lx, y: ly + 19, 'font-size': 10, fill: css('--muted') }, mode === 'diverging' ? `−${fmt(vmax)} (B방향)` : fmt(0)));
  svg.appendChild(el('text', { x: lx + lw, y: ly + 19, 'font-size': 10, fill: css('--muted'), 'text-anchor': 'end' }, mode === 'diverging' ? `+${fmt(vmax)} (A방향)` : fmt(vmax)));
  if (mode === 'diverging') svg.appendChild(el('text', { x: lx + lw / 2, y: ly + 19, 'font-size': 10, fill: css('--muted'), 'text-anchor': 'middle' }, '0'));
  container.appendChild(svg);
}

/** Horizontal bars, one hue (magnitude across categories) */
function barsH(container, { labels, values, fmt, color }) {
  container.innerHTML = '';
  const W = Math.max(container.clientWidth || 300, 200), labelW = 54, rowH = 22, H = labels.length * rowH + 8;
  const svg = svgEl(W, H); const vmax = Math.max(1e-9, ...values); const bw = W - labelW - 60;
  labels.forEach((lb, i) => {
    const y = 4 + i * rowH; const w = (values[i] / vmax) * bw;
    svg.appendChild(el('text', { x: labelW - 6, y: y + 14, 'text-anchor': 'end', 'font-size': 11, fill: css('--text-2') }, lb));
    const r = el('rect', { x: labelW, y: y + 3, width: Math.max(w, 0), height: rowH - 8, rx: 4, fill: color });
    r.addEventListener('mousemove', (ev) => showTip(`<div>${lb}: <b>${fmt(values[i])}</b></div>`, ev.clientX, ev.clientY)); r.addEventListener('mouseleave', hideTip);
    svg.appendChild(r);
    svg.appendChild(el('text', { x: labelW + w + 6, y: y + 14, 'font-size': 11, fill: css('--text-2') }, fmt(values[i])));
  });
  container.appendChild(svg);
}

/** Stacked bars of state shares per round — ordinal-diverging encoding on the B→A ladder */
function stateChart(container, rounds) {
  container.innerHTML = '';
  const order = ['B_strong', 'B_weak', 'undecided', 'A_weak', 'A_strong', 'abstain'];
  const colors = { B_strong: css('--series-2'), B_weak: mixHex(css('--series-2'), css('--surface-1'), 0.55), undecided: css('--axis'), A_weak: mixHex(css('--series-1'), css('--surface-1'), 0.55), A_strong: css('--series-1'), abstain: css('--grid') };
  const W = Math.max(container.clientWidth || 300, 240), H = 150, m = { l: 36, r: 8, t: 6, b: 22 }; const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const svg = svgEl(W, H); const n = rounds.length; const bw = iw / n;
  rounds.forEach((r, i) => {
    const total = Object.values(r.state_counts).reduce((a, b) => a + b, 0); let y = m.t + ih;
    order.forEach((k) => { const h = (r.state_counts[k] / total) * ih; y -= h; const rect = el('rect', { x: m.l + i * bw + 1, y: y + 1, width: Math.max(bw - 2, 1), height: Math.max(h - 2, 0), fill: colors[k] }); rect.addEventListener('mousemove', (ev) => showTip(`<div class="t">라운드 ${r.round_no}</div>` + order.map((kk) => `<div><span style="display:inline-block;width:10px;height:10px;border-radius:2px;background:${colors[kk]};margin-right:6px"></span>${STATE_KO[kk]}: <b>${fmtPct(r.state_counts[kk] / total)}</b></div>`).join(''), ev.clientX, ev.clientY)); rect.addEventListener('mouseleave', hideTip); rect.addEventListener('click', () => setRound(r.round_no)); svg.appendChild(rect); });
    if (i % Math.max(1, Math.ceil(n / 10)) === 0 || i === n - 1) svg.appendChild(el('text', { x: m.l + i * bw + bw / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 10, fill: css('--muted') }, 'R' + r.round_no));
  });
  [0, 0.5, 1].forEach((t) => svg.appendChild(el('text', { x: m.l - 4, y: m.t + ih - t * ih + 4, 'text-anchor': 'end', 'font-size': 10, fill: css('--muted') }, (t * 100) + '%')));
  const lg = document.createElement('div'); lg.className = 'legend'; lg.innerHTML = order.map((k) => `<span><i class="sw box" style="background:${colors[k]}"></i>${STATE_KO[k]}</span>`).join(''); container.appendChild(lg);
  container.appendChild(svg);
}

/* ───────────────────────── renderers ───────────────────────── */
function renderTrajectory() {
  const rs = state.run.rounds;
  lineChart($('#traj-chart'), { x: rs.map((r) => r.round_no), highlightX: state.round, series: [
    { name: '후보 A', color: css('--series-1'), values: rs.map((r) => r.support.A) },
    { name: '후보 B', color: css('--series-2'), values: rs.map((r) => r.support.B) }] });
  const goal = state.run.manifest.config.goal_margin;
  lineChart($('#margin-chart'), { x: rs.map((r) => r.round_no), height: 150, highlightX: state.round, refY: goal, refLabel: `목표 ${fmtPct(goal)}`, series: [
    { name: 'A − B', color: css('--series-1'), values: rs.map((r) => r.support.A - r.support.B) }] });
}
function renderCompare() {
  const c = state.run.compare; const wrap = $('#compare-wrap'); if (!c || !c.length) { wrap.hidden = true; return; } wrap.hidden = false;
  lineChart($('#compare-chart'), { x: c.map((r) => r.round), height: 200, highlightX: state.round, series: [
    { name: 'A · 방어 on', color: css('--series-1'), values: c.map((r) => r.A_defended) },
    { name: 'A · 방어 off', color: css('--series-2'), values: c.map((r) => r.A_undefended), dash: '5 4' }] });
}
function renderStateChart() { stateChart($('#state-chart'), state.run.rounds); }
function renderHeatmap() {
  const hm = state.run.heatmap; const box = $('#heatmap');
  if (!hm) { box.innerHTML = '<div class="muted">세그먼트 집계가 없습니다 (라운드 종료 후 생성).</div>'; return; }
  const metric = $('#hm-metric').value; const idx = Math.min(state.round, hm.rounds.length) - 1; if (idx < 0) return;
  const vals = hm[metric][idx];
  heatmap(box, { rows: hm.sido, cols: hm.age, values: vals, counts: hm.n, mode: metric === 'net_rate' ? 'diverging' : 'sequential', fmt: (v) => fmtPct(v), title: metric === 'net_rate' ? '누적 순이동률' : '노출률' });
}
function renderChannelBars() {
  const r = rec(); const sh = r.exposure.channel_share; const labels = Object.keys(sh);
  barsH($('#channel-bars'), { labels: labels.map((k) => CH_KO[k] || k), values: labels.map((k) => sh[k]), fmt: (v) => fmtPct(v), color: css('--series-1') });
}
function renderPlan() {
  const m = rec().manipulator;
  $('#plan-box').innerHTML = `<div><b>${m.frame_ko}</b> · ${m.claim_tier_ko} (허위도 ${m.falsehood})</div><div>${m.message}</div>
    <div>표적 셀 <b>${m.n_target_cells}</b>개 · 노출 <b>${fmtN(m.impressions_total)}</b> · 비용 <b>${fmtN(m.cost)}</b></div>
    <div class="muted">상위 셀: ${(m.target_cells_label_top5 || []).join(' · ')}</div>`;
}
function renderContrib() {
  const c = state.run.contribution; const box = $('#contrib-table');
  if (!c) { box.innerHTML = ''; return; }
  const lit = state.run.literacy;
  box.innerHTML = `<table><thead><tr><th>셀</th><th>n</th><th>노출</th><th>↑</th><th>↓</th><th>되돌림</th><th>순이동</th><th>순이동률</th><th>리터러시</th><th>설득가능성</th></tr></thead><tbody>` +
    c.map((r) => `<tr><td>${r.label}</td><td>${fmtN(r.n)}</td><td>${fmtN(r.exposed)}</td><td>${fmtN(r.moved_up)}</td><td>${fmtN(r.moved_down)}</td><td>${fmtN(r.reverted)}</td><td>${fmtN(r.net_up)}</td><td>${fmtPct(r.net_up_rate)}</td><td>${fmtP(r.mean_literacy, 2)}</td><td>${fmtP(r.mean_persuadability, 2)}</td></tr>`).join('') +
    `</tbody></table>` + (lit && lit.pearson_r != null ? `<div class="muted" style="padding:6px 8px">셀 단위 리터러시–순이동률 상관 r = ${fmtP(lit.pearson_r, 3)} (n_cells=${lit.n_cells})</div>` : '');
}
function renderBattleLog() {
  $('#battle-log').innerHTML = state.run.rounds.map((r) => { const m = r.manipulator, d = r.defense; return `<li${r.round_no === state.round ? ' style="color:var(--text-1)"' : ''}><span class="m">조작</span> [${m.frame_ko}/${m.claim_tier_ko}] 셀 ${m.n_target_cells} · 노출 ${fmtN(r.exposure.n_exposed)} · L1↑${fmtN(r.l1.n_up)} L2↑${r.l2.n_delta_up} L3↑${r.l3.n_moved_up} → <span class="${d.detected ? 'd' : 'miss'}">방어 ${d.detected ? `탐지(p=${fmtP(d.p_detect, 2)}) 정정노출 ${fmtN(d.n_corr_exposed)} 되돌림 ${fmtN(d.n_reverted)}` : `미탐지(p=${fmtP(d.p_detect, 2)})`}</span> · A ${fmtPct(r.support.A)}</li>`; }).join('');
}
function setRound(n) { state.round = n; state._roundPinned = n !== state.run.rounds.length; $('#hm-round').value = n; renderAll(); }

/* ───────────────────────── config form ───────────────────────── */
const FIELDS = { 'c-rounds': 'rounds', 'c-seed': 'seed', 'c-budget': 'budget', 'c-goal_margin': 'goal_margin', 'c-ethics_level': 'ethics_level',
  'c-defender_enabled': 'defender.enabled', 'c-defender_strength': 'defender.strength', 'c-defender_budget_ratio': 'defender.budget_ratio', 'c-defender_revert_base': 'defender.revert_base',
  'c-promotion_k': 'promotion.k', 'c-promotion_k_l3': 'promotion.k_l3', 'c-manipulator_n_target_cells': 'manipulator.n_target_cells', 'c-checkpoint_every': 'checkpoint_every',
  'c-backend_kind': 'backend.kind', 'c-backend_model': 'backend.model', 'c-backend_concurrency': 'backend.concurrency', 'c-backend_thinking_tokens': 'backend.thinking_tokens' };
const getPath = (o, p) => p.split('.').reduce((a, k) => (a == null ? undefined : a[k]), o);
async function loadConfig() {
  const c = await api('/api/config'); state.config = c.config;
  for (const id in FIELDS) { const e = document.getElementById(id); const v = getPath(c.config, FIELDS[id]); if (e.type === 'checkbox') e.checked = !!v; else if (v != null) e.value = v; }
  $('#v-ethics').textContent = $('#c-ethics_level').value; $('#v-dstr').textContent = $('#c-defender_strength').value;
  if (!$('#r-name').value) $('#r-name').value = (c.config.name || 'run') + '-' + new Date().toISOString().slice(5, 16).replace(/[-T:]/g, '');
}
function readOverrides() {
  const o = {};
  for (const id in FIELDS) { const e = document.getElementById(id); let v = e.type === 'checkbox' ? e.checked : e.value; if (v === '' || v == null) continue; if (e.type === 'number' || e.type === 'range') v = +v; o[FIELDS[id]] = v; }
  return o;
}
async function saveConfig() { try { const c = await api('/api/config', { method: 'PUT', body: JSON.stringify({ overrides: readOverrides() }) }); state.config = c.config; toast('설정 저장됨 (configs/campaign.yaml)'); if (state.fileKind === 'campaign') loadFile('campaign'); } catch (e) { toast('저장 실패: ' + e.message, true); } }
async function startRun(mode) {
  const name = $('#r-name').value.trim(); if (!name) { toast('실행 이름을 입력하세요', true); return; }
  const o = readOverrides();
  const body = { name, mode, resume: $('#r-resume').checked, overrides: o, data_dir: $('#r-data').value, backend_kind: o['backend.kind'] };
  try { const job = await api('/api/runs', { method: 'POST', body: JSON.stringify(body) }); toast(`작업 ${job.id} 대기열 등록 (${mode})`); attachJob(job.id); } catch (e) { toast(e.message, true); }
}
async function startBuild() {
  const body = { synthetic: $('#b-synthetic').value ? +$('#b-synthetic').value : null, limit: $('#b-limit').value ? +$('#b-limit').value : null, out: $('#b-out').value || null, seed: +$('#b-seed').value || null };
  try { const job = await api('/api/build', { method: 'POST', body: JSON.stringify(body) }); toast(`데이터 빌드 작업 ${job.id} 시작`); attachJob(job.id); } catch (e) { toast(e.message, true); }
}

/* ───────────────────────── files ───────────────────────── */
async function loadFile(kind) { state.fileKind = kind; $$('#file-tabs button').forEach((b) => b.classList.toggle('active', b.dataset.kind === kind)); const f = await api(`/api/files/${kind}`); $('#file-text').value = f.text; $('#file-msg').textContent = f.path; }
async function saveFile() { try { await api(`/api/files/${state.fileKind}`, { method: 'PUT', body: JSON.stringify({ text: $('#file-text').value }) }); $('#file-msg').textContent = '저장됨 ✓'; if (state.fileKind === 'campaign') loadConfig(); } catch (e) { $('#file-msg').textContent = '오류: ' + e.message; } }

/* ───────────────────────── jobs ───────────────────────── */
function attachJob(id) { state.jobId = id; state.jobLines = 0; $('#job-log').textContent = ''; $('#btn-stop').hidden = false; pollJob(); }
async function pollJob() {
  if (!state.jobId) return;
  let j; try { j = await api(`/api/jobs/${state.jobId}?since=${state.jobLines}`); } catch (e) { return; }
  if (j.lines.length) { const log = $('#job-log'); log.textContent += (log.textContent ? '\n' : '') + j.lines.join('\n'); log.scrollTop = log.scrollHeight; state.jobLines = j.n_lines; }
  const p = j.progress || {}; const bar = $('#job-progress .bar'), lab = $('#job-progress .label');
  const pct = p.rounds ? (p.round_no / p.rounds) * 100 : (j.status === 'done' ? 100 : 0);
  bar.style.width = pct + '%';
  lab.textContent = `${j.id} ${j.kind} ${j.params.name || p.out || ''} · ${j.status}${p.rounds ? ` · R${p.round_no}/${p.rounds} ${p.label || ''} A=${fmtP(p.A, 4)}` : ''}${j.error ? ' · ' + j.error : ''}`;
  const running = j.status === 'running' || j.status === 'queued';
  $('#btn-stop').hidden = !running;
  if (running && j.params.name && j.params.name === state.runName && Date.now() - state.lastRunReload > 4000 && p.round_no) { state.lastRunReload = Date.now(); state._roundPinned = false; loadRun(state.runName); }
  if (!running) { state.jobId = null; await loadStatus(); if (j.params.name) { $('#run-select').value = j.params.name; state._roundPinned = false; await loadRun(j.params.name); } }
}
async function stopJob() { if (state.jobId) { await api(`/api/jobs/${state.jobId}/stop`, { method: 'POST' }); toast('중지 요청 — 현재 라운드 종료 후 체크포인트 저장'); } }

/* ───────────────────────── report modal (minimal markdown) ───────────────────────── */
function mdToHtml(md) {
  const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;'); const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, '<b>$1</b>').replace(/`(.+?)`/g, '<code>$1</code>');
  const lines = md.split('\n'); let out = '', i = 0;
  while (i < lines.length) {
    const l = lines[i];
    if (/^\|/.test(l)) { const rows = []; while (i < lines.length && /^\|/.test(lines[i])) { rows.push(lines[i]); i++; } const cells = (r) => r.replace(/^\||\|$/g, '').split('|').map((c) => c.trim()); const head = cells(rows[0]); const body = rows.slice(2); out += `<table><thead><tr>${head.map((h) => `<th>${inline(h)}</th>`).join('')}</tr></thead><tbody>${body.map((r) => `<tr>${cells(r).map((c) => `<td>${inline(c)}</td>`).join('')}</tr>`).join('')}</tbody></table>`; continue; }
    if (/^# /.test(l)) out += `<h1>${inline(l.slice(2))}</h1>`; else if (/^## /.test(l)) out += `<h2>${inline(l.slice(3))}</h2>`; else if (/^### /.test(l)) out += `<h3>${inline(l.slice(4))}</h3>`;
    else if (/^- /.test(l)) { let items = ''; while (i < lines.length && /^- /.test(lines[i])) { items += `<li>${inline(lines[i].slice(2))}</li>`; i++; } out += `<ul>${items}</ul>`; continue; }
    else if (l.trim()) out += `<p>${inline(l)}</p>`;
    i++;
  }
  return out;
}
async function showReport() {
  if (!state.runName) return;
  const md = await api(`/api/runs/${encodeURIComponent(state.runName)}/report?display=${$('#rep-display').checked ? 1 : 0}`);
  $('#modal-title').textContent = `리포트 — ${state.runName}`; $('#modal-body').innerHTML = mdToHtml(md) + `<p class="muted">다운로드: <a href="/api/runs/${state.runName}/download/rounds.jsonl${state.run.sub ? '?sub=' + state.run.sub : ''}">rounds.jsonl</a> · <a href="/api/runs/${state.runName}/download/segments.parquet${state.run.sub ? '?sub=' + state.run.sub : ''}">segments.parquet</a> · <a href="/api/runs/${state.runName}/download/l2_responses.jsonl${state.run.sub ? '?sub=' + state.run.sub : ''}">l2_responses.jsonl</a></p>`; $('#modal').hidden = false;
}
async function deleteRun() { if (!state.runName) return; if (!confirm(`runs/${state.runName} 을(를) 삭제할까요? 되돌릴 수 없습니다.`)) return; try { await api(`/api/runs/${encodeURIComponent(state.runName)}`, { method: 'DELETE' }); state.runName = null; state.run = null; await loadStatus(); } catch (e) { toast(e.message, true); } }

/* ───────────────────────── wiring ───────────────────────── */
function bind() {
  $('#btn-refresh').onclick = () => { loadStatus(); if (state.runName) loadRun(state.runName); };
  $('#run-select').onchange = (e) => { state._roundPinned = false; loadRun(e.target.value); };
  $('#btn-build').onclick = startBuild; $('#btn-save').onclick = saveConfig; $('#btn-run').onclick = () => startRun('run'); $('#btn-compare').onclick = () => startRun('compare'); $('#btn-stop').onclick = stopJob;
  $('#btn-file-save').onclick = saveFile; $$('#file-tabs button').forEach((b) => (b.onclick = () => loadFile(b.dataset.kind)));
  $('#btn-report').onclick = showReport; $('#btn-delete').onclick = deleteRun; $('#modal-close').onclick = () => ($('#modal').hidden = true); $('#modal').onclick = (e) => { if (e.target.id === 'modal') $('#modal').hidden = true; };
  $('#c-ethics_level').oninput = (e) => ($('#v-ethics').textContent = e.target.value); $('#c-defender_strength').oninput = (e) => ($('#v-dstr').textContent = e.target.value);
  $('#hm-round').oninput = (e) => setRound(+e.target.value); $('#hm-metric').onchange = renderHeatmap;
  $('#hm-play').onclick = () => { if (state.playing) { clearInterval(state.playing); state.playing = null; $('#hm-play').textContent = '▶'; return; } $('#hm-play').textContent = '⏸'; state.playing = setInterval(() => { if (!state.run) return; const n = state.run.rounds.length; setRound(state.round >= n ? 1 : state.round + 1); }, 700); };
  $('#btn-theme').onclick = () => { const cur = document.documentElement.dataset.theme || (matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'); const next = cur === 'dark' ? 'light' : 'dark'; document.documentElement.dataset.theme = next; try { localStorage.setItem('of_theme', next); } catch (_) { } renderAll(); };
  let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(renderAll, 150); });
}
(async function init() {
  try { const t = localStorage.getItem('of_theme'); if (t) document.documentElement.dataset.theme = t; } catch (_) { }
  bind();
  await loadStatus(); await loadConfig(); await loadFile('campaign');
  setInterval(() => { loadStatus().catch(() => { }); if (state.jobId) pollJob(); }, 2500);
})();
