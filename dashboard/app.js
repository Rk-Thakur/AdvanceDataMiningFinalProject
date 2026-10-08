/* Interactive wine classification dashboard.
 * Reads data.js (results, written by wine_classification.py) and interactive_data.js
 * (written by export_interactive_data.py). Plain SVG/HTML, no libraries, works offline.
 *
 * Interactive features:
 *   depth slider · decision-threshold explorer · predict-a-wine · model toggles · metric picker
 *   Laplace α selector · SVM kernel/C selector · linked keyword highlight · minimum-review filter
 *   animated transitions (respects prefers-reduced-motion)
 */
(() => {
  'use strict';

  const D = window.WINE_RESULTS;
  const I = window.WINE_INTERACTIVE;
  if (!D || !I) {
    document.getElementById('app-error').hidden = false;
    return;
  }

  const MODELS = ['Naive Bayes', 'Decision Tree', 'SVM'];
  const MODEL_VAR = { 'Naive Bayes': '--s-nb', 'Decision Tree': '--s-dt', 'SVM': '--s-svm' };
  const NS = 'http://www.w3.org/2000/svg';
  const FEATS = I.features;
  const KW = new Map(D.keywords.all.map((r) => [r.keyword, r]));
  const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ------------------------------------------------------------------ utils
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const modelColor = (m) => css(MODEL_VAR[m]);
  const pct = (v, d = 2) => (v * 100).toFixed(d) + '%';
  const pct1 = (v) => pct(v, 1);
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const std = (a) => { const m = mean(a); return Math.sqrt(mean(a.map((v) => (v - m) ** 2))); };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const label = (m) => (m === 'Decision Tree' ? `Decision Tree (depth ${state.depth})` : D.models[m].label);

  function setAttrs(e, attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') e.setAttribute('class', v);
      else if (k === 'text') e.textContent = v;
      else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
      else e.setAttribute(k, v === true ? '' : v);
    }
  }
  function append(e, kids) {
    kids.flat(Infinity).forEach((k) => {
      if (k == null || k === false) return;
      e.append(k instanceof Node ? k : document.createTextNode(String(k)));
    });
    return e;
  }
  const h = (tag, attrs = {}, ...kids) => { const e = document.createElement(tag); setAttrs(e, attrs); return append(e, kids); };
  const s = (tag, attrs = {}, ...kids) => { const e = document.createElementNS(NS, tag); setAttrs(e, attrs); return append(e, kids); };

  const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
  function ticks(min, max, count = 5) {
    const step0 = (max - min) / count;
    const mag = 10 ** Math.floor(Math.log10(step0));
    const err = step0 / mag;
    const step = (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1) * mag;
    const out = [];
    for (let v = Math.ceil(min / step - 1e-9) * step; v <= max + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const textWidth = (str, size = 11) => String(str).length * size * 0.6;

  function vBarPath(x, yBase, yEnd, w, r = 4) {
    const top = Math.min(yBase, yEnd), bot = Math.max(yBase, yEnd), ht = bot - top;
    if (ht <= 0.01 || w <= 0) return '';
    r = Math.min(r, w / 2, ht);
    if (yEnd <= yBase) return `M${x},${bot}V${top + r}Q${x},${top} ${x + r},${top}H${x + w - r}Q${x + w},${top} ${x + w},${top + r}V${bot}Z`;
    return `M${x},${top}V${bot - r}Q${x},${bot} ${x + r},${bot}H${x + w - r}Q${x + w},${bot} ${x + w},${bot - r}V${top}Z`;
  }
  function hBarPath(xBase, xEnd, y, ht, r = 4, roundEnd = true) {
    const left = Math.min(xBase, xEnd), right = Math.max(xBase, xEnd), w = right - left;
    if (w <= 0.01 || ht <= 0) return '';
    r = roundEnd ? Math.min(r, ht / 2, w) : 0;
    if (xEnd >= xBase) return `M${left},${y}H${right - r}Q${right},${y} ${right},${y + r}V${y + ht - r}Q${right},${y + ht} ${right - r},${y + ht}H${left}Z`;
    return `M${right},${y}H${left + r}Q${left},${y} ${left},${y + r}V${y + ht - r}Q${left},${y + ht} ${left + r},${y + ht}H${right}Z`;
  }

  // classification metrics from confusion counts
  function scores(c) {
    const n = c.TP + c.FP + c.FN + c.TN;
    const prec = c.TP + c.FP ? c.TP / (c.TP + c.FP) : 0;
    const rec = c.TP + c.FN ? c.TP / (c.TP + c.FN) : 0;
    return {
      acc: (c.TP + c.TN) / n, prec, rec,
      spec: c.TN + c.FP ? c.TN / (c.TN + c.FP) : 0,
      f1: prec + rec ? (2 * prec * rec) / (prec + rec) : 0,
    };
  }

  // ---------------------------------------------------------------- theme
  function isDark() {
    const t = document.documentElement.getAttribute('data-theme');
    if (t) return t === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  }
  const RAMP_LIGHT = ['#e3eefc', '#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab'];
  const RAMP_DARK = ['#1d2633', '#1a3150', '#173d6c', '#184f95', '#1c5cab', '#256abf', '#2a78d6', '#3987e5', '#5598e7', '#6da7ec', '#86b6ef'];
  function seq(t) {
    const ramp = isDark() ? RAMP_DARK : RAMP_LIGHT;
    const i = clamp(Math.round(t * (ramp.length - 1)), 0, ramp.length - 1);
    const ink = isDark() ? (i >= 8 ? '#0b0b0b' : '#ffffff') : (i >= 6 ? '#ffffff' : '#0b0b0b');
    return { bg: ramp[i], ink };
  }

  // -------------------------------------------------------------- tooltip
  const tip = h('div', { class: 'tip', role: 'tooltip' });
  document.body.append(tip);
  function showTip(evt, title, rows) {
    tip.replaceChildren();
    if (title) tip.append(h('div', { class: 'tip-title', text: title }));
    rows.forEach((r) => {
      tip.append(h('div', { class: 'tip-row' },
        r.color ? h('span', { class: 'tip-key', style: `background:${r.color}` }) : null,
        h('strong', { text: r.value }),
        r.label ? h('span', { class: 'tip-label', text: r.label }) : null));
    });
    tip.classList.add('on');
    placeTip(evt);
  }
  function placeTip(evt) {
    let x, y;
    if (evt && evt.clientX != null && evt.type !== 'focus' && evt.type !== 'keydown') { x = evt.clientX; y = evt.clientY; }
    else { const b = evt.target.getBoundingClientRect(); x = b.left + b.width / 2; y = b.top; }
    const w = tip.offsetWidth, ht = tip.offsetHeight;
    let left = x + 14, top = y - ht - 12;
    if (left + w > innerWidth - 8) left = x - w - 14;
    if (left < 8) left = 8;
    if (top < 8) top = y + 18;
    tip.style.left = left + 'px';
    tip.style.top = top + 'px';
  }
  const hideTip = () => tip.classList.remove('on');
  function bindTip(hit, title, rows, mark) {
    setAttrs(hit, { tabindex: '0', class: ((hit.getAttribute('class') || '') + ' hit').trim() });
    const on = (e) => { showTip(e, title, typeof rows === 'function' ? rows() : rows); if (mark) mark.classList.add('lift'); };
    const off = () => { hideTip(); if (mark) mark.classList.remove('lift'); };
    hit.addEventListener('pointermove', on);
    hit.addEventListener('pointerleave', off);
    hit.addEventListener('focus', on);
    hit.addEventListener('blur', off);
  }
  function bindClick(node, fn) {
    node.classList.add('clickable');
    node.addEventListener('click', fn);
    node.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fn(); } });
  }

  // ------------------------------------------------------- state + charts
  const state = {
    depth: D.dt.bestDepth,
    thrModel: 'Naive Bayes',
    thr: { 'Naive Bayes': 0.5, SVM: 0 },
    visible: new Set(MODELS),
    metric: 'all',
    alpha: 1,
    kernel: 'linear',
    C: 1,
    keyword: null,
    minCount: D.keywords.minCount,
    predict: { kw: new Set(), wine: null, modified: false, text: '' },
  };

  /* A chart is redrawn from scratch. When it has a data() function, state changes tween
     from the current values to the new ones (numbers interpolate; labels switch). */
  let charts = [];
  let hooks = [];
  function chart(container, draw, opts = {}) {
    const c = { container, draw, data: opts.data || null, tags: opts.tags || [], width: container.clientWidth, cur: null, raf: 0 };
    c.cur = c.data ? c.data() : null;
    charts.push(c);
    draw(container, c.cur);
    return c;
  }
  const onState = (tags, fn) => hooks.push({ tags, fn });
  function lerp(a, b, t) {
    if (typeof a === 'number' && typeof b === 'number') return a + (b - a) * t;
    if (Array.isArray(a) && Array.isArray(b) && a.length === b.length) return b.map((v, i) => lerp(a[i], v, t));
    if (a && b && typeof a === 'object' && typeof b === 'object' && !Array.isArray(a) && !Array.isArray(b)) {
      const o = {};
      for (const k of Object.keys(b)) o[k] = lerp(a[k], b[k], t);
      return o;
    }
    return b;
  }
  function redraw(c) { c.container.replaceChildren(); c.draw(c.container, c.cur); }
  function animate(c) {
    cancelAnimationFrame(c.raf);
    clearTimeout(c.timer);
    const next = c.data ? c.data() : null;
    if (!c.data || reduceMotion || c.cur == null || document.hidden) { c.cur = next; redraw(c); return; }
    const from = c.cur;
    const t0 = performance.now();
    const dur = 450;
    const finish = () => { cancelAnimationFrame(c.raf); c.cur = next; redraw(c); };
    const step = (now) => {
      const t = Math.min(1, (now - t0) / dur);
      if (t >= 1) { clearTimeout(c.timer); finish(); return; }
      c.cur = lerp(from, next, 1 - (1 - t) ** 3);
      redraw(c);
      c.raf = requestAnimationFrame(step);
    };
    c.raf = requestAnimationFrame(step);
    // if animation frames are paused (background tab), still land on the final values
    c.timer = setTimeout(finish, dur + 150);
  }
  function update(...tags) {
    hideTip();
    charts.forEach((c) => { if (c.tags.some((t) => tags.includes(t))) animate(c); });
    hooks.forEach((k) => { if (k.tags.some((t) => tags.includes(t))) k.fn(); });
  }
  function resizeAll() {
    hideTip();
    charts.forEach((c) => {
      const w = c.container.clientWidth;
      if (w === c.width) return;
      c.width = w;
      redraw(c);
    });
  }

  // ------------------------------------------------------- UI components
  function card(parent, { title, sub, wide = false }) {
    const body = h('div', { class: 'card-body' });
    const head = h('figcaption', {}, h('h3', { text: title }), sub ? h('p', { class: 'sub', text: sub }) : null);
    const el = h('figure', { class: 'card' + (wide ? ' wide' : '') }, head, body);
    parent.append(el);
    return { el, body, head };
  }

  function dataTable(cols, rows, { sortable = false, highlight, rowClick, rowSelected } = {}) {
    const table = h('table');
    const thead = h('thead');
    const tbody = h('tbody');
    let sortKey = null, sortDir = -1;
    const draw = () => {
      tbody.replaceChildren();
      const data = rows.slice();
      if (sortKey) {
        const col = cols.find((c) => c.key === sortKey);
        data.sort((a, b) => {
          const va = col.sortVal ? col.sortVal(a) : a[sortKey];
          const vb = col.sortVal ? col.sortVal(b) : b[sortKey];
          if (typeof va === 'string') return sortDir * va.localeCompare(vb) * -1;
          return sortDir * ((va ?? -Infinity) - (vb ?? -Infinity));
        });
      }
      data.forEach((r) => {
        const cls = [highlight && highlight(r) ? 'hl' : '', rowSelected && rowSelected(r) ? 'sel' : ''].join(' ').trim();
        const tr = h('tr', { class: cls || null });
        cols.forEach((c) => {
          const v = r[c.key];
          const tdCls = [c.best && c.best(r) ? 'best' : '', c.selected && c.selected(r) ? 'cell-sel' : ''].join(' ').trim();
          const td = h('td', { class: tdCls || null });
          if (c.render) append(td, [c.render(r)]);
          else td.textContent = v == null ? '—' : (c.fmt ? c.fmt(v, r) : String(v));
          if (c.style) { const st = c.style(r); if (st) td.setAttribute('style', st); }
          tr.append(td);
        });
        if (rowClick) { tr.setAttribute('tabindex', '0'); bindClick(tr, () => rowClick(r)); }
        tbody.append(tr);
      });
    };
    const tr = h('tr');
    cols.forEach((c) => {
      const th = h('th', { text: c.label, scope: 'col', class: sortable ? 'sortable' : null });
      if (sortable) {
        th.addEventListener('click', () => {
          if (sortKey === c.key) sortDir *= -1; else { sortKey = c.key; sortDir = -1; }
          draw();
        });
      }
      tr.append(th);
    });
    thead.append(tr);
    table.append(thead, tbody);
    draw();
    table.redraw = draw;
    table.setRows = (r) => { rows = r; draw(); };
    return table;
  }

  function addDataView(c, cols, rows) {
    const holder = h('div', { class: 'table-wrap' });
    const d = h('details', { class: 'dataview' }, h('summary', { text: 'View data table' }), holder);
    const fill = () => holder.replaceChildren(dataTable(cols, typeof rows === 'function' ? rows() : rows));
    fill();
    c.el.append(d);
    return fill;
  }

  function legend(items) {
    return h('div', { class: 'legend' }, items.map((i) =>
      h('span', { class: 'lg' }, h('span', { class: 'sw ' + (i.kind || ''), style: `--c:${i.color}` }), i.name)));
  }

  function kpi(parent, { label: lbl, value, sub, color }) {
    parent.append(h('div', { class: 'kpi', style: `--i:${parent.children.length}` },
      h('div', { class: 'label' }, color ? h('span', { class: 'key', style: `background:${color}` }) : null, lbl),
      h('div', { class: 'value', text: value }),
      sub ? h('div', { class: 'sub', text: sub }) : null));
  }

  function segmented(name, options, value, onChange) {
    const g = h('div', { class: 'seg', role: 'radiogroup', 'aria-label': name });
    options.forEach((o) => {
      const on = o.value === value;
      const b = h('button', { type: 'button', role: 'radio', 'aria-checked': String(on), class: on ? 'on' : null, text: o.label });
      b.addEventListener('click', () => {
        g.querySelectorAll('button').forEach((x) => { x.classList.remove('on'); x.setAttribute('aria-checked', 'false'); });
        b.classList.add('on');
        b.setAttribute('aria-checked', 'true');
        onChange(o.value);
      });
      g.append(b);
    });
    return g;
  }
  const ctl = (lbl, ...kids) => h('div', { class: 'ctl' }, h('span', { class: 'ctl-label' }, lbl), ...kids);

  /* slider over a list of discrete values, with tick labels */
  function stepSlider({ values, value, onInput, aria, fmt = String }) {
    const input = h('input', { type: 'range', min: 0, max: values.length - 1, step: 1, value: values.indexOf(value), 'aria-label': aria });
    const tickRow = h('div', { class: 'range-ticks' }, values.map((v) => h('span', { text: fmt(v) })));
    const mark = () => [...tickRow.children].forEach((t, i) => t.classList.toggle('on', i === +input.value));
    input.addEventListener('input', () => { mark(); onInput(values[+input.value]); });
    input.setAttribute('aria-valuetext', fmt(value));
    mark();
    return h('div', {}, input, tickRow);
  }

  function svgRoot(W, H, aria) {
    return s('svg', { class: 'chart', width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': aria });
  }
  function yAxis(svg, y, tks, m, W, fmt, title) {
    tks.forEach((t) => {
      svg.append(s('line', { class: 'grid-line', x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }));
      svg.append(s('text', { class: 'tick', x: m.l - 8, y: y(t), 'text-anchor': 'end', 'dominant-baseline': 'middle', text: fmt(t) }));
    });
    if (title) {
      const cy = (m.t + y(tks[0])) / 2;
      svg.append(s('text', { class: 'axis-title', x: 12, y: cy, transform: `rotate(-90 12 ${cy})`, 'text-anchor': 'middle', text: title }));
    }
  }

  // --------------------------------------------------------------- charts
  function groupedBars(container, o) {
    const { groups, series, yMin = 0, yMax, yFmt, valFmt, height = 260, yTitle, refLine, xTitle, aria, highlightGroup } = o;
    const W = Math.max(container.clientWidth, 260);
    const m = { t: 14, r: 12, b: xTitle ? 46 : 30, l: yTitle ? 60 : 46 };
    const baseB = m.b;
    const gwEst = (W - m.l - m.r) / groups.length;
    const maxLbl = Math.max(...groups.map((g) => textWidth(g.label)));
    const rotate = gwEst >= 24 && maxLbl > gwEst - 6;
    if (rotate) m.b += Math.min(50, maxLbl * 0.6);
    const H = height + (m.b - baseB);
    const svg = svgRoot(W, H, aria || 'Bar chart');
    const y = lin(yMin, yMax, H - m.b, m.t);
    yAxis(svg, y, ticks(yMin, yMax, 5), m, W, yFmt, yTitle);
    const gw = (W - m.l - m.r) / groups.length;
    const inner = Math.min(gw * 0.74, series.length * 44);
    const bw = Math.max(2, (inner - (series.length - 1) * 2) / series.length);
    const every = gw < 24 ? Math.ceil(24 / gw) : 1;
    groups.forEach((g, gi) => {
      const gx = m.l + gi * gw + (gw - inner) / 2;
      const dim = highlightGroup != null && highlightGroup !== gi;
      g.values.forEach((v, si) => {
        const x = gx + si * (bw + 2);
        svg.append(s('path', { class: 'mark grow-v', style: `--i:${gi * series.length + si}`, d: vBarPath(x, y(yMin), y(Math.max(v, yMin)), bw), fill: series[si].color, opacity: dim ? 0.35 : null }));
      });
      const hit = s('rect', { x: m.l + gi * gw, y: m.t, width: gw, height: H - m.b - m.t, fill: 'transparent' });
      bindTip(hit, g.title || g.label, series.map((se, si) => ({ label: se.name, value: valFmt(g.values[si]), color: se.color })));
      if (g.onClick) bindClick(hit, g.onClick);
      svg.append(hit);
      if (gi % every === 0) {
        const lx = m.l + gi * gw + gw / 2, ly = H - m.b + 16;
        const attrs = rotate
          ? { class: 'cat', x: lx, y: ly - 4, 'text-anchor': 'end', transform: `rotate(-35 ${lx} ${ly - 4})`, text: g.label }
          : { class: 'cat', x: lx, y: ly, 'text-anchor': 'middle', text: g.label };
        if (highlightGroup === gi) attrs['font-weight'] = '700';
        svg.append(s('text', attrs));
      }
    });
    svg.append(s('line', { class: 'axis-line', x1: m.l, x2: W - m.r, y1: y(yMin), y2: y(yMin) }));
    if (refLine) {
      svg.append(s('line', { class: 'ref-line', x1: m.l, x2: W - m.r, y1: y(refLine.value), y2: y(refLine.value) }));
      svg.append(s('text', { class: 'val strong', x: W - m.r, y: y(refLine.value) - 6, 'text-anchor': 'end', text: refLine.label }));
    }
    if (xTitle) svg.append(s('text', { class: 'axis-title', x: m.l + (W - m.l - m.r) / 2, y: H - 6, 'text-anchor': 'middle', text: xTitle }));
    if (series.length > 1) container.append(legend(series));
    container.append(svg);
  }

  /* Ranked horizontal bars. rows: [{label, value, color, tip?, key?}] - rows with a keyword
     key are clickable and take part in the linked keyword highlight. */
  function hBars(container, o) {
    const { rows, fmt, xMin, xMax, rowH = 22, aria, xTitle, valueLabels = true } = o;
    const W = Math.max(container.clientWidth, 260);
    if (!rows.length) { container.append(h('p', { class: 'empty', text: o.emptyText || 'Nothing to show.' })); return; }
    const labelW = Math.min(170, Math.max(...rows.map((r) => textWidth(r.label))) + 12);
    const lo = xMin ?? Math.min(0, ...rows.map((r) => r.value));
    const hi = xMax ?? Math.max(0, ...rows.map((r) => r.value));
    const m = { t: 6, r: valueLabels ? 52 : 14, b: xTitle ? 40 : 24, l: labelW };
    const H = m.t + m.b + rows.length * rowH;
    const svg = svgRoot(W, H, aria || 'Horizontal bar chart');
    const x = lin(lo, hi, m.l + (lo < 0 ? 40 : 0), W - m.r);
    // bars start at 0 when it is on the axis, otherwise at the axis edge; ends stay inside the plot
    const base = clamp(0, lo, hi);
    const xv = (v) => x(clamp(v, lo, hi));
    const tickFmt = o.tickFmt || fmt;
    ticks(lo, hi, Math.max(2, Math.floor((W - m.l) / 90))).forEach((t) => {
      svg.append(s('line', { class: 'grid-line', x1: x(t), x2: x(t), y1: m.t, y2: H - m.b }));
      svg.append(s('text', { class: 'tick', x: x(t), y: H - m.b + 14, 'text-anchor': 'middle', text: tickFmt(t) }));
    });
    const sel = state.keyword;
    rows.forEach((r, i) => {
      const yy = m.t + i * rowH;
      const isSel = r.key && sel === r.key;
      const dim = sel && r.key !== undefined && !isSel;
      const p = s('path', { class: 'mark ' + (r.value < base ? 'grow-hn' : 'grow-h'), style: `--i:${i}`, d: hBarPath(x(base), xv(r.value), yy + 3, rowH - 6), fill: r.color, opacity: dim ? 0.3 : null });
      svg.append(p);
      svg.append(s('text', {
        class: 'cat', x: m.l - 8, y: yy + rowH / 2, 'text-anchor': 'end', 'dominant-baseline': 'middle', text: r.label,
        'font-weight': isSel ? '700' : null, opacity: dim ? 0.55 : null,
      }));
      if (valueLabels) {
        const neg = r.value < base;
        svg.append(s('text', { class: 'val' + (isSel ? ' strong' : ''), x: xv(r.value) + (neg ? -6 : 6), y: yy + rowH / 2, 'text-anchor': neg ? 'end' : 'start', 'dominant-baseline': 'middle', text: fmt(r.value), opacity: dim ? 0.55 : null }));
      }
      const hit = s('rect', { x: 0, y: yy, width: W, height: rowH, fill: 'transparent' });
      bindTip(hit, r.label, r.tip || [{ value: fmt(r.value), color: r.color }], p);
      if (r.key) bindClick(hit, () => selectKeyword(r.key));
      svg.append(hit);
    });
    svg.append(s('line', { class: 'axis-line', x1: x(base), x2: x(base), y1: m.t, y2: H - m.b }));
    if (xTitle) svg.append(s('text', { class: 'axis-title', x: m.l + (W - m.l - m.r) / 2, y: H - 6, 'text-anchor': 'middle', text: xTitle }));
    if (o.legend) container.append(legend(o.legend));
    container.append(svg);
  }

  function stackedHBars(container, o) {
    const { rows, series, fmt, rowH = 22, aria, xTitle } = o;
    const W = Math.max(container.clientWidth, 260);
    const labelW = Math.min(170, Math.max(...rows.map((r) => textWidth(r.label))) + 12);
    const hi = Math.max(...rows.map((r) => r.parts.reduce((a, b) => a + b, 0)));
    const m = { t: 6, r: 44, b: xTitle ? 40 : 24, l: labelW };
    const H = m.t + m.b + rows.length * rowH;
    const svg = svgRoot(W, H, aria || 'Stacked bar chart');
    const x = lin(0, hi, m.l, W - m.r);
    ticks(0, hi, Math.max(2, Math.floor((W - m.l) / 90))).forEach((t) => {
      svg.append(s('line', { class: 'grid-line', x1: x(t), x2: x(t), y1: m.t, y2: H - m.b }));
      svg.append(s('text', { class: 'tick', x: x(t), y: H - m.b + 14, 'text-anchor': 'middle', text: fmt(t) }));
    });
    const sel = state.keyword;
    rows.forEach((r, i) => {
      const yy = m.t + i * rowH;
      let acc = 0;
      const total = r.parts.reduce((a, b) => a + b, 0);
      const isSel = sel === r.key;
      const dim = sel && !isSel;
      const marks = [];
      r.parts.forEach((v, si) => {
        if (v <= 0) return;
        const last = r.parts.slice(si + 1).every((q) => q <= 0);
        const x0 = x(acc) + (acc > 0 ? 2 : 0);
        const p = s('path', { class: 'mark grow-h', style: `--i:${i}`, d: hBarPath(x0, x(acc + v), yy + 3, rowH - 6, 4, last), fill: series[si].color, opacity: dim ? 0.3 : null });
        svg.append(p);
        marks.push(p);
        acc += v;
      });
      svg.append(s('text', { class: 'cat', x: m.l - 8, y: yy + rowH / 2, 'text-anchor': 'end', 'dominant-baseline': 'middle', text: r.label, 'font-weight': isSel ? '700' : null, opacity: dim ? 0.55 : null }));
      svg.append(s('text', { class: 'val', x: x(total) + 6, y: yy + rowH / 2, 'dominant-baseline': 'middle', text: fmt(total), opacity: dim ? 0.55 : null }));
      const hit = s('rect', { x: 0, y: yy, width: W, height: rowH, fill: 'transparent' });
      bindTip(hit, r.label, [{ label: 'total', value: fmt(total) }].concat(
        series.map((se, si) => ({ label: se.name, value: fmt(r.parts[si]), color: se.color }))));
      hit.addEventListener('pointerenter', () => marks.forEach((p) => p.classList.add('lift')));
      hit.addEventListener('pointerleave', () => marks.forEach((p) => p.classList.remove('lift')));
      if (r.key) bindClick(hit, () => selectKeyword(r.key));
      svg.append(hit);
    });
    if (xTitle) svg.append(s('text', { class: 'axis-title', x: m.l + (W - m.l - m.r) / 2, y: H - 6, 'text-anchor': 'middle', text: xTitle }));
    container.append(legend(series));
    container.append(svg);
  }

  function lineChart(container, o) {
    const { xs, series, yMin, yMax, yFmt, valFmt, xTitle, yTitle, height = 260, aria, markers = true, highlightX, onPick } = o;
    const W = Math.max(container.clientWidth, 260);
    const H = height;
    const endLabelW = Math.max(...series.map((se) => textWidth(se.name))) + 14;
    const m = { t: 14, r: series.length > 1 ? Math.min(endLabelW, 110) : 16, b: xTitle ? 46 : 30, l: yTitle ? 60 : 46 };
    const svg = svgRoot(W, H, aria || 'Line chart');
    const y = lin(yMin, yMax, H - m.b, m.t);
    const pad = 18;
    const x = lin(0, xs.length - 1, m.l + pad, W - m.r - pad);
    yAxis(svg, y, ticks(yMin, yMax, 5), m, W, yFmt, yTitle);
    if (highlightX != null) {
      svg.append(s('rect', { x: x(highlightX) - 14, y: m.t, width: 28, height: H - m.b - m.t, fill: css('--surface-2') }));
    }
    const spacing = xs.length > 1 ? x(1) - x(0) : 100;
    const everyX = Math.max(1, Math.ceil((Math.max(...xs.map((v) => textWidth(v))) + 10) / spacing));
    xs.forEach((xv, i) => {
      if (i % everyX !== 0 && i !== highlightX) return;
      if (i !== highlightX && highlightX != null && Math.abs(i - highlightX) < everyX) return;
      svg.append(s('text', {
        class: 'cat', x: x(i), y: H - m.b + 16, 'text-anchor': 'middle', text: String(xv), 'font-weight': highlightX === i ? '700' : null,
      }));
    });
    svg.append(s('line', { class: 'axis-line', x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b }));
    series.forEach((se) => {
      if (se.band) {
        const top = se.band.map((b, i) => `${x(i)},${y(b[1])}`).join(' ');
        const bot = se.band.map((b, i) => `${x(i)},${y(b[0])}`).reverse().join(' ');
        svg.append(s('polygon', { class: 'fade', points: `${top} ${bot}`, fill: se.color, opacity: 0.12 }));
      }
    });
    const endLabels = [];
    series.forEach((se) => {
      const d = se.values.map((v, i) => `${i ? 'L' : 'M'}${x(i)},${y(v)}`).join('');
      svg.append(s('path', {
        d, fill: 'none', stroke: se.color, 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round',
        'stroke-dasharray': se.kind === 'dash' ? '6 4' : se.kind === 'dot' ? '2 4' : null,
        class: se.kind ? 'fade' : 'draw', pathLength: se.kind ? null : 1,
      }));
      if (markers) se.values.forEach((v, i) => svg.append(s('circle', {
        class: 'pop', style: `--i:${i}`, cx: x(i), cy: y(v), r: highlightX === i ? 6 : 4, fill: se.color, stroke: css('--surface'), 'stroke-width': 2,
      })));
      endLabels.push({ y: y(se.values[se.values.length - 1]), name: se.name });
    });
    endLabels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < endLabels.length; i++) if (endLabels[i].y - endLabels[i - 1].y < 13) endLabels[i].y = endLabels[i - 1].y + 13;
    if (series.length > 1) endLabels.forEach((l) => svg.append(s('text', { class: 'val', x: W - m.r + 8, y: l.y, 'dominant-baseline': 'middle', text: l.name })));

    const cross = s('line', { class: 'crosshair', y1: m.t, y2: H - m.b, visibility: 'hidden' });
    svg.append(cross);
    const overlay = s('rect', { x: m.l, y: m.t, width: W - m.l - m.r, height: H - m.t - m.b, fill: 'transparent', tabindex: '0', class: 'hit' + (onPick ? ' clickable' : '') });
    let idx = highlightX ?? 0;
    const show = (e) => {
      cross.setAttribute('x1', x(idx)); cross.setAttribute('x2', x(idx)); cross.setAttribute('visibility', 'visible');
      showTip(e, (o.xLabel || '') + xs[idx], series.map((se) => ({ label: se.name, value: valFmt(se.values[idx], se, idx), color: se.color })));
    };
    const nearest = (e) => {
      const r = svg.getBoundingClientRect();
      const px = e.clientX - r.left;
      let best = 0;
      xs.forEach((_, i) => { if (Math.abs(x(i) - px) < Math.abs(x(best) - px)) best = i; });
      return best;
    };
    overlay.addEventListener('pointermove', (e) => { idx = nearest(e); show(e); });
    overlay.addEventListener('focus', (e) => show(e));
    overlay.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') idx = Math.min(xs.length - 1, idx + 1);
      else if (e.key === 'ArrowLeft') idx = Math.max(0, idx - 1);
      else if ((e.key === 'Enter' || e.key === ' ') && onPick) { e.preventDefault(); onPick(idx); return; }
      else return;
      e.preventDefault();
      show({ target: overlay, type: 'keydown' });
    });
    if (onPick) overlay.addEventListener('click', (e) => onPick(nearest(e)));
    const off = () => { cross.setAttribute('visibility', 'hidden'); hideTip(); };
    overlay.addEventListener('pointerleave', off);
    overlay.addEventListener('blur', off);
    svg.append(overlay);
    if (xTitle) svg.append(s('text', { class: 'axis-title', x: m.l + (W - m.l - m.r) / 2, y: H - 6, 'text-anchor': 'middle', text: xTitle }));
    if (series.length > 1) container.append(legend(series.map((se) => ({ name: se.name, color: se.color, kind: se.kind || 'line' }))));
    container.append(svg);
  }

  /* ROC curves for the given models; optional point marks the current threshold. */
  function rocChart(container, { models, point }) {
    const W = Math.max(container.clientWidth, 260);
    const H = Math.min(W, 420);
    const m = { t: 12, r: 16, b: 44, l: 52 };
    const svg = svgRoot(W, H, 'ROC curves');
    const plotW = Math.min(W - m.l - m.r, H - m.t - m.b);
    const x = lin(0, 1, m.l, m.l + plotW);
    const y = lin(0, 1, H - m.b, H - m.b - plotW);
    ticks(0, 1, 5).forEach((t) => {
      svg.append(s('line', { class: 'grid-line', x1: x(0), x2: x(1), y1: y(t), y2: y(t) }));
      svg.append(s('line', { class: 'grid-line', x1: x(t), x2: x(t), y1: y(0), y2: y(1) }));
      svg.append(s('text', { class: 'tick', x: m.l - 8, y: y(t), 'text-anchor': 'end', 'dominant-baseline': 'middle', text: t.toFixed(1) }));
      svg.append(s('text', { class: 'tick', x: x(t), y: H - m.b + 14, 'text-anchor': 'middle', text: t.toFixed(1) }));
    });
    svg.append(s('line', { class: 'ref-line', x1: x(0), y1: y(0), x2: x(1), y2: y(1) }));
    svg.append(s('text', { class: 'axis-title', x: x(0.5), y: H - 8, 'text-anchor': 'middle', text: 'False positive rate (1 − specificity)' }));
    svg.append(s('text', { class: 'axis-title', x: 14, y: y(0.5), transform: `rotate(-90 14 ${y(0.5)})`, 'text-anchor': 'middle', text: 'True positive rate (sensitivity)' }));
    models.forEach((mName) => {
      svg.append(s('path', {
        d: D.models[mName].roc.map((p, i) => `${i ? 'L' : 'M'}${x(p[0])},${y(p[1])}`).join(''),
        fill: 'none', stroke: modelColor(mName), 'stroke-width': 2, 'stroke-linejoin': 'round', class: 'draw', pathLength: 1,
      }));
    });
    const tprAt = (pts, f) => { let b = 0; pts.forEach((p) => { if (p[0] <= f + 1e-9) b = Math.max(b, p[1]); }); return b; };
    const cross = s('line', { class: 'crosshair', y1: y(1), y2: y(0), visibility: 'hidden' });
    svg.append(cross);
    if (point) {
      svg.append(s('line', { class: 'ref-line', x1: x(point.fpr), x2: x(point.fpr), y1: y(0), y2: y(point.tpr) }));
      svg.append(s('line', { class: 'ref-line', x1: x(0), x2: x(point.fpr), y1: y(point.tpr), y2: y(point.tpr) }));
      svg.append(s('circle', { cx: x(point.fpr), cy: y(point.tpr), r: 7, fill: point.color, stroke: css('--surface'), 'stroke-width': 2 }));
      const right = x(point.fpr) < m.l + plotW * 0.6;
      svg.append(s('text', { class: 'val strong', x: x(point.fpr) + (right ? 12 : -12), y: y(point.tpr) + 18, 'text-anchor': right ? 'start' : 'end', text: `threshold ${point.label}` }));
    }
    const overlay = s('rect', { x: x(0), y: y(1), width: plotW, height: plotW, fill: 'transparent', tabindex: '0', class: 'hit' });
    let f = 0.1;
    const show = (e) => {
      cross.setAttribute('x1', x(f)); cross.setAttribute('x2', x(f)); cross.setAttribute('visibility', 'visible');
      showTip(e, `False positive rate ${f.toFixed(2)}`, models.map((mName) => ({
        label: `${label(mName)} TPR`, value: tprAt(D.models[mName].roc, f).toFixed(3), color: modelColor(mName),
      })));
    };
    overlay.addEventListener('pointermove', (e) => { const r = svg.getBoundingClientRect(); f = clamp((e.clientX - r.left - m.l) / plotW, 0, 1); show(e); });
    overlay.addEventListener('focus', show);
    overlay.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight') f = Math.min(1, f + 0.05);
      else if (e.key === 'ArrowLeft') f = Math.max(0, f - 0.05);
      else return;
      e.preventDefault();
      show({ target: overlay, type: 'keydown' });
    });
    const off = () => { cross.setAttribute('visibility', 'hidden'); hideTip(); };
    overlay.addEventListener('pointerleave', off);
    overlay.addEventListener('blur', off);
    svg.append(overlay);
    container.append(legend(models.map((mName) => ({ name: `${D.models[mName].label} · AUC ${D.models[mName].auc.toFixed(3)}`, color: modelColor(mName), kind: 'line' }))));
    container.append(svg);
  }

  function cvStrip(container, models) {
    const W = Math.max(container.clientWidth, 260);
    const H = 280;
    const m = { t: 14, r: 14, b: 34, l: 52 };
    const all = MODELS.flatMap((mm) => D.models[mm].cv);
    const lo = Math.floor(Math.min(...all) * 20) / 20 - 0.02;
    const hi = Math.min(1, Math.ceil(Math.max(...all) * 20) / 20 + 0.02);
    const svg = svgRoot(W, H, '5-fold cross-validation accuracy by model');
    const y = lin(lo, hi, H - m.b, m.t);
    yAxis(svg, y, ticks(lo, hi, 5), m, W, (v) => pct(v, 0));
    const gw = (W - m.l - m.r) / models.length;
    models.forEach((mName, gi) => {
      const cx = m.l + gi * gw + gw / 2;
      const cv = D.models[mName].cv;
      const col = modelColor(mName);
      const mu = mean(cv);
      svg.append(s('line', { x1: cx - 26, x2: cx + 26, y1: y(mu), y2: y(mu), stroke: col, 'stroke-width': 3, 'stroke-linecap': 'round' }));
      svg.append(s('text', { class: 'val strong', x: cx + 32, y: y(mu), 'dominant-baseline': 'middle', text: pct(mu) }));
      cv.forEach((v, i) => {
        const dx = (i - 2) * 9;
        const dot = s('circle', { class: 'mark pop', style: `--i:${gi * 5 + i}`, cx: cx + dx, cy: y(v), r: 5, fill: col, stroke: css('--surface'), 'stroke-width': 2 });
        svg.append(dot);
        const hit = s('circle', { cx: cx + dx, cy: y(v), r: 12, fill: 'transparent' });
        bindTip(hit, `${D.models[mName].label} · fold ${i + 1}`, [{ value: pct(v), label: 'accuracy', color: col }], dot);
        svg.append(hit);
      });
      svg.append(s('text', { class: 'cat', x: cx, y: H - m.b + 16, 'text-anchor': 'middle', text: D.models[mName].label }));
    });
    container.append(svg);
  }

  function keywordScatter(container, minCount) {
    const rows = D.keywords.all.filter((r) => r.all >= minCount);
    const W = Math.max(container.clientWidth, 260);
    const H = Math.min(460, Math.max(320, W * 0.55));
    const m = { t: 14, r: 20, b: 44, l: 56 };
    const hi = Math.ceil(Math.max(...rows.map((r) => Math.max(r.pctPos, r.pctNeg))) / 5) * 5;
    const svg = svgRoot(W, H, 'Keyword frequency in 90+ versus 90- wines');
    const x = lin(0, hi, m.l, W - m.r);
    const y = lin(0, hi, H - m.b, m.t);
    ticks(0, hi, 6).forEach((t) => {
      svg.append(s('line', { class: 'grid-line', x1: m.l, x2: W - m.r, y1: y(t), y2: y(t) }));
      svg.append(s('line', { class: 'grid-line', x1: x(t), x2: x(t), y1: m.t, y2: H - m.b }));
      svg.append(s('text', { class: 'tick', x: m.l - 8, y: y(t), 'text-anchor': 'end', 'dominant-baseline': 'middle', text: t + '%' }));
      svg.append(s('text', { class: 'tick', x: x(t), y: H - m.b + 14, 'text-anchor': 'middle', text: t + '%' }));
    });
    svg.append(s('line', { class: 'ref-line', x1: x(0), y1: y(0), x2: x(hi), y2: y(hi) }));
    svg.append(s('text', { class: 'axis-title', x: m.l + (W - m.l - m.r) / 2, y: H - 8, 'text-anchor': 'middle', text: '% of 90- wines containing the keyword' }));
    svg.append(s('text', { class: 'axis-title', x: 14, y: (m.t + H - m.b) / 2, transform: `rotate(-90 14 ${(m.t + H - m.b) / 2})`, 'text-anchor': 'middle', text: '% of 90+ wines containing the keyword' }));
    const pos = css('--pos'), neg = css('--neg');
    const gap = (r) => r.pctPos - r.pctNeg;
    const sel = state.keyword;
    const labelled = new Set(rows.slice().sort((a, b) => Math.abs(gap(b)) - Math.abs(gap(a))).slice(0, sel ? 8 : 12).map((r) => r.keyword));
    if (sel) labelled.add(sel);
    rows.forEach((r, ri) => {
      const isSel = r.keyword === sel;
      svg.append(s('circle', {
        class: 'mark pop', style: `--i:${Math.min(ri, 60)}`, cx: x(r.pctNeg), cy: y(r.pctPos), r: isSel ? 7 : 4, fill: gap(r) >= 0 ? pos : neg,
        stroke: isSel ? css('--text') : css('--surface'), 'stroke-width': isSel ? 2.5 : 1.5, opacity: sel && !isSel ? 0.45 : null,
      }));
    });
    const placed = [];
    rows.filter((r) => labelled.has(r.keyword)).sort((a, b) => (a.keyword === sel ? -1 : b.keyword === sel ? 1 : 0)).forEach((r) => {
      let ly = y(r.pctPos) - 9;
      const lx = x(r.pctNeg) + 7;
      while (placed.some((p) => Math.abs(p.y - ly) < 12 && Math.abs(p.x - lx) < 90)) ly -= 12;
      placed.push({ x: lx, y: ly });
      const anchorEnd = lx > W - 110;
      svg.append(s('text', { class: 'val' + (r.keyword === sel ? ' strong' : ''), x: anchorEnd ? x(r.pctNeg) - 7 : lx, y: ly, 'text-anchor': anchorEnd ? 'end' : 'start', text: r.keyword }));
    });
    rows.forEach((r) => {
      const hit = s('circle', { cx: x(r.pctNeg), cy: y(r.pctPos), r: 12, fill: 'transparent' });
      bindTip(hit, r.keyword, [
        { value: r.pctPos.toFixed(1) + '%', label: 'of 90+ wines', color: pos },
        { value: r.pctNeg.toFixed(1) + '%', label: 'of 90- wines', color: neg },
        { value: r.p90 == null ? '—' : pct1(r.p90), label: 'of wines with it are 90+' },
      ]);
      bindClick(hit, () => selectKeyword(r.keyword));
      svg.append(hit);
    });
    container.append(legend([{ name: 'More common in 90+ wines', color: pos }, { name: 'More common in 90- wines', color: neg }]));
    container.append(svg);
    if (sel && !rows.some((r) => r.keyword === sel)) {
      container.append(h('p', { class: 'note', text: `${sel} appears in fewer than ${minCount} reviews, so it is hidden here. Lower the minimum-review filter to see it.` }));
    }
  }

  /* Confusion matrix; counts may be fractional mid-animation, so they are rounded for display. */
  function confusion(container, t, title) {
    const c = { TP: Math.round(t.TP), FP: Math.round(t.FP), FN: Math.round(t.FN), TN: Math.round(t.TN) };
    const max = Math.max(t.TP, t.TN, t.FP, t.FN) || 1;
    const box = h('div');
    if (title) box.append(h('h4', { text: title }));
    const grid = h('div', { class: 'cm', role: 'table', 'aria-label': (title || '') + ' confusion matrix' });
    let ci = 0;
    const cell = (raw, n, k, rowTotal) => {
      const col = seq(raw / max);
      return h('div', { class: 'cell pop-cell', role: 'cell', style: `background:${col.bg};color:${col.ink};--i:${ci++}` },
        h('div', { class: 'n', text: n }),
        h('div', { class: 'k', text: `${k} · ${rowTotal ? pct1(n / rowTotal) : '—'} of row` }));
    };
    const negTotal = c.TN + c.FP, posTotal = c.FN + c.TP;
    grid.append(
      h('div', { class: 'hd' }),
      h('div', { class: 'hd', text: 'Predicted 90-' }),
      h('div', { class: 'hd', text: 'Predicted 90+' }),
      h('div', { class: 'hd row', text: 'Actual 90-' }), cell(t.TN, c.TN, 'TN', negTotal), cell(t.FP, c.FP, 'FP', negTotal),
      h('div', { class: 'hd row', text: 'Actual 90+' }), cell(t.FN, c.FN, 'FN', posTotal), cell(t.TP, c.TP, 'TP', posTotal));
    box.append(grid);
    container.append(box);
  }

  /* Distribution of model scores on the test set by real class, with the threshold. */
  function scoreHistogram(container, { model, thr }) {
    const isNB = model === 'Naive Bayes';
    const vals = I.test.map((w) => (isNB ? w.nb : w.svm));
    const lo = isNB ? 0 : Math.floor(Math.min(...vals));
    const hiV = isNB ? 1 : Math.ceil(Math.max(...vals));
    const bins = isNB ? 20 : Math.round((hiV - lo) / 0.25);
    const bw = (hiV - lo) / bins;
    const hist = Array.from({ length: bins }, (_, i) => ({ x0: lo + i * bw, pos: 0, neg: 0 }));
    I.test.forEach((w, i) => {
      const b = clamp(Math.floor((vals[i] - lo) / bw), 0, bins - 1);
      hist[b][w.label ? 'pos' : 'neg']++;
    });
    const W = Math.max(container.clientWidth, 260);
    const H = 270;
    const m = { t: 18, r: 14, b: 44, l: 46 };
    const svg = svgRoot(W, H, 'Score distribution by real grade');
    const maxC = Math.max(...hist.map((b) => Math.max(b.pos, b.neg)));
    const yMax = Math.max(5, Math.ceil(maxC / 10) * 10);
    const y = lin(0, yMax, H - m.b, m.t);
    const x = lin(lo, hiV, m.l, W - m.r);
    // region predicted 90+
    svg.append(s('rect', { x: x(thr), y: m.t, width: Math.max(0, x(hiV) - x(thr)), height: H - m.b - m.t, fill: css('--surface-2') }));
    yAxis(svg, y, ticks(0, yMax, 4), m, W, (v) => v, null);
    const slot = x(lo + bw) - x(lo);
    const barW = Math.max(1, (slot - 4) / 2);
    hist.forEach((b) => {
      const bx = x(b.x0) + 1;
      svg.append(s('path', { class: 'mark grow-v', d: vBarPath(bx, y(0), y(b.pos), barW), fill: css('--pos') }));
      svg.append(s('path', { class: 'mark grow-v', d: vBarPath(bx + barW + 2, y(0), y(b.neg), barW), fill: css('--neg') }));
      const hit = s('rect', { x: x(b.x0), y: m.t, width: slot, height: H - m.b - m.t, fill: 'transparent' });
      const range = isNB ? `${pct(b.x0, 0)}–${pct(b.x0 + bw, 0)}` : `${b.x0.toFixed(2)} to ${(b.x0 + bw).toFixed(2)}`;
      bindTip(hit, `Score ${range}`, [
        { value: `${b.pos} wines`, label: 'real 90+', color: css('--pos') },
        { value: `${b.neg} wines`, label: 'real 90-', color: css('--neg') },
      ]);
      svg.append(hit);
    });
    ticks(lo, hiV, isNB ? 5 : 6).forEach((t) => svg.append(s('text', { class: 'tick', x: x(t), y: H - m.b + 14, 'text-anchor': 'middle', text: isNB ? pct(t, 0) : t })));
    svg.append(s('line', { class: 'axis-line', x1: m.l, x2: W - m.r, y1: y(0), y2: y(0) }));
    svg.append(s('line', { x1: x(thr), x2: x(thr), y1: m.t - 6, y2: H - m.b, stroke: css('--text'), 'stroke-width': 2 }));
    const right = x(thr) < W - 140;
    svg.append(s('text', { class: 'val strong', x: x(thr) + (right ? 6 : -6), y: m.t + 4, 'text-anchor': right ? 'start' : 'end', text: `threshold ${isNB ? pct(thr, 0) : thr.toFixed(2)} → predict 90+` }));
    svg.append(s('text', { class: 'axis-title', x: m.l + (W - m.l - m.r) / 2, y: H - 6, 'text-anchor': 'middle', text: isNB ? 'Naive Bayes P(90+)' : 'SVM decision score (w·x + b)' }));
    container.append(legend([{ name: 'Real 90+', color: css('--pos') }, { name: 'Real 90-', color: css('--neg') }]));
    container.append(svg);
  }

  // ------------------------------------------------- shared derived data
  const ds = D.dataset;
  const depthRow = (d) => D.dt.depths.find((r) => r.depth === d);
  /* test counts for each model; the decision tree follows the depth slider */
  function testCounts(mName) {
    if (mName === 'Decision Tree') return I.trees[state.depth].counts;
    const t = D.models[mName].test;
    return { TP: t.TP, FP: t.FP, FN: t.FN, TN: t.TN };
  }
  function modelMetrics(mName) {
    if (mName === 'Decision Tree') {
      const r = depthRow(state.depth);
      return { acc: r.test, prec: r.precision, rec: r.recall, spec: r.specificity, f1: r.f1, cv: r.cvMean, cvs: r.cvStd, auc: D.models[mName].auc };
    }
    const t = D.models[mName].test;
    return { acc: t.Accuracy, prec: t.Precision, rec: t['Recall (Sensitivity)'], spec: t.Specificity, f1: t['F1-score'], cv: mean(D.models[mName].cv), cvs: std(D.models[mName].cv), auc: D.models[mName].auc };
  }
  const visibleModels = () => MODELS.filter((m) => state.visible.has(m));

  function thresholdCounts(mName, thr) {
    const c = { TP: 0, FP: 0, FN: 0, TN: 0 };
    I.test.forEach((w) => {
      const p = (mName === 'SVM' ? w.svm : w.nb) > thr ? 1 : 0;
      if (p && w.label) c.TP++; else if (p) c.FP++; else if (w.label) c.FN++; else c.TN++;
    });
    return c;
  }

  // ------------------------------------------------- keyword selection
  const inspector = document.getElementById('inspector');
  function selectKeyword(k) {
    state.keyword = state.keyword === k ? null : k;
    update('keyword');
  }
  function renderInspector() {
    const k = state.keyword;
    inspector.hidden = !k;
    inspector.replaceChildren();
    if (!k) return;
    const r = KW.get(k);
    const j = FEATS.indexOf(k);
    const nbRatio = Math.log(I.nb.p1[j][1] / I.nb.p1[j][0]);
    const dtImp = I.trees[state.depth].importance.findIndex((q) => q.keyword === k);
    const close = h('button', { class: 'ins-close', type: 'button', 'aria-label': 'Clear keyword selection', text: '×' });
    close.addEventListener('click', () => selectKeyword(k));
    const goScatter = h('button', { class: 'btn', type: 'button', text: 'Show in scatter' });
    goScatter.addEventListener('click', () => document.getElementById('kw-scatter').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'center' }));
    const tryIt = h('button', { class: 'btn', type: 'button', text: 'Add to "Predict a wine"' });
    tryIt.addEventListener('click', () => { state.predict.kw.add(j); state.predict.modified = true; update('predict'); document.getElementById('predict').scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth' }); });
    inspector.append(
      h('div', { class: 'ins-head' }, h('span', { class: 'ins-title', text: k }), close),
      h('dl', {},
        h('dt', { text: 'Reviews containing it' }), h('dd', { text: `${r.all} (${r.pos} 90+, ${r.neg} 90-)` }),
        h('dt', { text: 'Rated 90+ when present' }), h('dd', { text: r.p90 == null ? '—' : pct1(r.p90) }),
        h('dt', { text: 'Naive Bayes log-ratio' }), h('dd', { text: nbRatio.toFixed(2) }),
        h('dt', { text: 'SVM weight' }), h('dd', { text: I.svm.weights[j].toFixed(2) }),
        h('dt', { text: `Tree importance (depth ${state.depth})` }), h('dd', { text: dtImp >= 0 ? `#${dtImp + 1} of top 15` : 'not in top 15' })),
      h('div', { class: 'ins-actions' }, goScatter, tryIt));
  }

  // ============================================================ sections
  const bestOverall = MODELS.reduce((a, b) => (D.models[b].test.Accuracy > D.models[a].test.Accuracy ? b : a));
  const bestCvModel = MODELS.reduce((a, b) => (mean(D.models[b].cv) > mean(D.models[a].cv) ? b : a));

  function metricKpis(parent, mName) {
    const t = D.models[mName].test;
    const cv = D.models[mName].cv;
    kpi(parent, { label: 'Test accuracy', value: pct(t.Accuracy), sub: `${t.TP + t.TN} of ${ds.test} correct`, color: modelColor(mName) });
    kpi(parent, { label: '5-fold CV accuracy', value: pct(mean(cv)), sub: `± ${pct(std(cv))} across folds` });
    kpi(parent, { label: 'Precision', value: pct(t.Precision), sub: 'predicted 90+ that are 90+' });
    kpi(parent, { label: 'Recall (sensitivity)', value: pct(t['Recall (Sensitivity)']), sub: '90+ wines found' });
    kpi(parent, { label: 'Specificity', value: pct(t.Specificity), sub: '90- wines found' });
    kpi(parent, { label: 'F1-score', value: pct(t['F1-score']) });
    kpi(parent, { label: 'ROC AUC', value: D.models[mName].auc.toFixed(3) });
  }

  function cvFoldCard(grid, mName) {
    const cv = D.models[mName].cv;
    const c = card(grid, { title: '5-fold cross-validation', sub: `Accuracy on each held-out fold of all ${ds.wines} wines; dashed line is the mean.` });
    const lo = Math.max(0, Math.floor((Math.min(...cv) - 0.1) * 10) / 10);
    chart(c.body, (el) => groupedBars(el, {
      groups: cv.map((v, i) => ({ label: `Fold ${i + 1}`, values: [v] })),
      series: [{ name: 'Accuracy', color: modelColor(mName) }],
      yMin: lo, yMax: 1, yFmt: (v) => pct(v, 0), valFmt: (v) => pct(v),
      refLine: { value: mean(cv), label: `mean ${pct(mean(cv))}` }, aria: `${mName} cross-validation accuracy per fold`,
    }));
    addDataView(c, [{ key: 'fold', label: 'Fold' }, { key: 'acc', label: 'Accuracy', fmt: (v) => pct(v) }],
      cv.map((v, i) => ({ fold: i + 1, acc: v })).concat([{ fold: 'Mean', acc: mean(cv) }]));
  }

  function divergingKeywordCard(grid, { title, sub, rows, fmt, xTitle }) {
    const c = card(grid, { title, sub });
    chart(c.body, (el) => hBars(el, {
      rows: rows.slice().sort((a, b) => b.value - a.value).map((r) => ({
        label: r.keyword, key: r.keyword, value: r.value, color: r.value >= 0 ? css('--pos') : css('--neg'),
        tip: [{ value: fmt(r.value), label: r.value >= 0 ? 'points to 90+' : 'points to 90-', color: r.value >= 0 ? css('--pos') : css('--neg') }, { value: 'click', label: 'to highlight everywhere' }],
      })),
      fmt, xTitle,
      legend: [{ name: 'Indicates 90+', color: css('--pos') }, { name: 'Indicates 90-', color: css('--neg') }],
      aria: title,
    }), { tags: ['keyword'] });
    addDataView(c, [{ key: 'keyword', label: 'Keyword' }, { key: 'value', label: 'Value', fmt }], rows.slice().sort((a, b) => b.value - a.value));
  }

  // ----------------------------------------------------------- overview
  function buildOverview() {
    document.getElementById('meta').textContent =
      `${ds.wines} wines from ${ds.file} · ${ds.attributes} keyword attributes · ` +
      `${Math.round((1 - ds.testSize) * 100)}/${Math.round(ds.testSize * 100)} stratified split (seed ${ds.seed}) · ${ds.folds}-fold cross-validation.`;
    document.getElementById('overview-lead').textContent =
      `Each wine is a binary vector of ${ds.attributes} Computational Wine Wheel attributes extracted from its Wine Spectator review. ` +
      `The label is 1 for a score of 90 or higher and 0 otherwise. Classes are almost balanced (${ds.pos} vs ${ds.neg}), so accuracy is a fair headline metric.`;

    const link = (href, text) => h('a', { href, text });
    document.getElementById('whatsnew').append(h('h3', { text: 'What you can change on this page' }), h('ul', {},
      h('li', {}, link('#dt', 'Depth slider'), ' — pick a tree depth 5–9 and every decision-tree number, chart and the confusion matrix follows it.'),
      h('li', {}, link('#threshold', 'Threshold explorer'), ' — move the decision threshold and watch TP, FP, FN and TN trade off live.'),
      h('li', {}, link('#predict', 'Predict a wine'), ' — load a test wine, paste a tasting note or tick keywords; all three models predict instantly.'),
      h('li', {}, link('#compare', 'Model toggles and metric picker'), ', ', link('#nb', 'Laplace α selector'), ', ', link('#svm', 'kernel and C selector'), '.'),
      h('li', {}, 'Click any keyword in any chart to highlight it everywhere; ', link('#knowledge', 'filter keywords by minimum reviews'), '. Changes animate.')));

    const f = document.getElementById('findings');
    const dtBest = depthRow(D.dt.bestDepth);
    const kwTop = D.keywords.pos.slice(0, 5).map((k) => k.keyword.toLowerCase()).join(', ');
    f.append(h('h3', { text: 'Key findings' }), h('ul', {},
      h('li', {}, h('strong', { text: D.models[bestOverall].label }), ` has the highest test accuracy (${pct(D.models[bestOverall].test.Accuracy)}); `,
        h('strong', { text: D.models[bestCvModel].label }), ` has the highest 5-fold CV accuracy (${pct(mean(D.models[bestCvModel].cv))}).`),
      h('li', {}, `Naive Bayes and SVM are within ${pct(Math.abs(D.models['Naive Bayes'].test.Accuracy - D.models.SVM.test.Accuracy))} of each other; the decision tree trails by more than ${Math.floor((Math.min(D.models['Naive Bayes'].test.Accuracy, D.models.SVM.test.Accuracy) - D.models['Decision Tree'].test.Accuracy) * 100)} points.`),
      h('li', {}, `Deeper trees fit the training set better (${pct1(D.dt.depths[0].train)} → ${pct1(D.dt.depths[D.dt.depths.length - 1].train)}) but test accuracy stays between ${pct1(Math.min(...D.dt.depths.map((d) => d.test)))} and ${pct1(Math.max(...D.dt.depths.map((d) => d.test)))}; best depth by CV is ${D.dt.bestDepth} (${pct1(dtBest.cvMean)}).`),
      h('li', {}, `Reviews of 90+ wines use more descriptors, and words like ${kwTop} appear almost only in 90+ reviews.`)));

    const k = document.getElementById('overview-kpis');
    kpi(k, { label: 'Wines', value: ds.wines.toLocaleString(), sub: ds.file });
    kpi(k, { label: 'Keyword attributes', value: ds.attributes, sub: 'binary (present / absent)' });
    kpi(k, { label: 'Training / testing', value: `${ds.train} / ${ds.test}`, sub: `stratified, seed ${ds.seed}` });
    kpi(k, { label: '90+ share', value: pct1(ds.pos / ds.wines), sub: `${ds.pos} of ${ds.wines} wines`, color: css('--pos') });
    kpi(k, { label: 'Best test accuracy', value: pct(D.models[bestOverall].test.Accuracy), sub: D.models[bestOverall].label, color: modelColor(bestOverall) });

    const g = document.getElementById('overview-grid');
    const c1 = card(g, { title: 'Class balance in each split', sub: 'Number of wines rated 90+ and 90- in the full data, training and testing sets.' });
    const splits = [
      { label: 'Full data', pos: ds.pos, neg: ds.neg },
      { label: 'Training', pos: ds.trainPos, neg: ds.trainNeg },
      { label: 'Testing', pos: ds.testPos, neg: ds.testNeg },
    ];
    chart(c1.body, (el) => groupedBars(el, {
      groups: splits.map((sp) => ({ label: sp.label, values: [sp.pos, sp.neg] })),
      series: [{ name: '90+ (1)', color: css('--pos') }, { name: '90- (0)', color: css('--neg') }],
      yMin: 0, yMax: Math.ceil(ds.pos / 100) * 100, yFmt: (v) => v, valFmt: (v) => `${v} wines`, yTitle: 'Wines', aria: 'Class balance per split',
    }));
    addDataView(c1, [{ key: 'label', label: 'Set' }, { key: 'pos', label: '90+' }, { key: 'neg', label: '90-' }], splits);

    const c2 = card(g, { title: 'Keywords per review', sub: 'How many wheel attributes each review contains, by class.' });
    chart(c2.body, (el) => groupedBars(el, {
      groups: ds.kwHist.map((r) => ({ label: String(r.n), title: `${r.n} keywords`, values: [r.pos, r.neg] })),
      series: [{ name: '90+ (1)', color: css('--pos') }, { name: '90- (0)', color: css('--neg') }],
      yMin: 0, yMax: Math.ceil(Math.max(...ds.kwHist.map((r) => Math.max(r.pos, r.neg))) / 20) * 20,
      yFmt: (v) => v, valFmt: (v) => `${v} wines`, xTitle: 'Keywords in the review', yTitle: 'Wines', aria: 'Keywords per review histogram',
    }));
    addDataView(c2, [{ key: 'n', label: 'Keywords' }, { key: 'pos', label: '90+' }, { key: 'neg', label: '90-' }], ds.kwHist);
  }

  // -------------------------------------------------------- A) Naive Bayes
  function buildNB() {
    const t = D.models['Naive Bayes'].test;
    document.getElementById('nb-lead').textContent =
      `Bernoulli Naive Bayes written from scratch: P(class | review) ∝ P(class) · Π P(keyword | class), with Laplace smoothing (α = 1) ` +
      `so an unseen keyword never zeroes out a class. It correctly labels ${t.TP + t.TN} of ${ds.test} test wines and matches scikit-learn's BernoulliNB exactly.`;
    metricKpis(document.getElementById('nb-kpis'), 'Naive Bayes');
    const g = document.getElementById('nb-grid');

    const c1 = card(g, { title: 'Confusion matrix', sub: `Testing set, ${ds.test} wines.` });
    confusion(c1.body, t);
    cvFoldCard(g, 'Naive Bayes');

    // Laplace α selector (#6)
    const al = D.nb.alpha;
    const base = al.find((a) => a.alpha === 1).cv;
    const c3 = card(g, { title: 'Effect of Laplace smoothing', sub: 'Pick α (or click the chart) to compare its 5-fold CV accuracy with the α = 1 used by the model.' });
    const ctrlHolder = h('div', { class: 'controls in-card' });
    c3.body.append(ctrlHolder);
    const segHolder = h('div');
    const drawSeg = () => segHolder.replaceChildren(segmented('Laplace smoothing alpha', al.map((a) => ({ value: a.alpha, label: String(a.alpha) })), state.alpha, (v) => { state.alpha = v; update('alpha'); }));
    drawSeg();
    ctrlHolder.append(ctl('Smoothing α', segHolder));
    const readout = h('p', { class: 'readout' });
    const renderReadout = () => {
      const cur = al.find((a) => a.alpha === state.alpha).cv;
      const d = cur - base;
      readout.replaceChildren();
      append(readout, [
        `α = ${state.alpha}: `, h('strong', { text: pct(cur) }), ' CV accuracy',
        state.alpha === 1 ? ' (the value used by the model).'
          : [' — ', h('span', { class: d >= 0 ? 'delta-up' : 'delta-down', text: `${d >= 0 ? '+' : '−'}${Math.abs(d * 100).toFixed(2)} pts` }), ' vs α = 1.']]);
    };
    renderReadout();
    const plot = h('div');
    c3.body.append(plot, readout);
    const lo = Math.floor((Math.min(...al.map((a) => a.cv)) - 0.02) * 50) / 50;
    const hi = Math.ceil((Math.max(...al.map((a) => a.cv)) + 0.01) * 50) / 50;
    chart(plot, (el) => lineChart(el, {
      xs: al.map((a) => a.alpha), xLabel: 'α = ',
      series: [{ name: 'CV accuracy', color: modelColor('Naive Bayes'), values: al.map((a) => a.cv) }],
      yMin: lo, yMax: hi, yFmt: (v) => pct(v, 0), valFmt: (v) => pct(v), xTitle: 'Laplace smoothing α', aria: 'Naive Bayes accuracy by smoothing',
      highlightX: al.findIndex((a) => a.alpha === state.alpha),
      onPick: (i) => { state.alpha = al[i].alpha; drawSeg(); update('alpha'); },
    }), { tags: ['alpha'] });
    onState(['alpha'], renderReadout);
    addDataView(c3, [{ key: 'alpha', label: 'α' }, { key: 'cv', label: 'CV accuracy', fmt: (v) => pct(v) }], al);

    divergingKeywordCard(g, {
      title: 'Most indicative keywords',
      sub: 'log P(keyword | 90+) / P(keyword | 90-). Click a keyword to highlight it across the page.',
      rows: D.nb.topKeywords, fmt: (v) => v.toFixed(2), xTitle: 'Log-likelihood ratio',
    });
  }

  // ------------------------------------------------------ B) Decision Tree
  function buildDT() {
    const dts = D.dt.depths;
    const bd = D.dt.bestDepth;
    const col = modelColor('Decision Tree');
    const lead = document.getElementById('dt-lead');
    const renderLead = () => {
      const r = depthRow(state.depth);
      lead.textContent =
        `Trees grown with entropy / information gain (ID3/C4.5 criterion), maximum depth 5–9. Depth ${bd} has the best cross-validation accuracy. ` +
        `Showing depth ${state.depth}: ${pct(r.test)} test accuracy, ${r.leaves} leaves, and a ${pct1(r.train - r.test)} gap between training and testing accuracy.`;
    };
    renderLead();
    onState(['depth'], renderLead);

    // depth slider (#1)
    document.getElementById('dt-controls').append(h('div', { class: 'controls' },
      h('div', { class: 'ctl grow' }, h('span', { class: 'ctl-label' }, 'Maximum tree depth'),
        stepSlider({ values: dts.map((d) => d.depth), value: state.depth, aria: 'Maximum tree depth', fmt: (v) => (v === bd ? `${v} (best CV)` : String(v)), onInput: (v) => { state.depth = v; update('depth'); } })),
      h('p', { class: 'note', style: 'margin:0;max-width:420px', text: 'Every card in this section, the comparison section and "Predict a wine" use the selected depth.' })));

    chart(document.getElementById('dt-kpis'), (el, d) => {
      kpi(el, { label: 'Selected depth', value: String(state.depth), sub: state.depth === bd ? 'best by cross-validation' : `best by CV is ${bd}`, color: col });
      kpi(el, { label: 'Test accuracy', value: pct(d.test), sub: `${Math.round(d.TP + d.TN)} of ${ds.test} correct` });
      kpi(el, { label: '5-fold CV accuracy', value: pct(d.cvMean), sub: `± ${pct(d.cvStd)}` });
      kpi(el, { label: 'Training accuracy', value: pct(d.train) });
      kpi(el, { label: 'Train − test gap', value: pct1(d.train - d.test), sub: 'overfitting indicator' });
      kpi(el, { label: 'Leaves', value: String(Math.round(d.leaves)) });
    }, { tags: ['depth'], data: () => Object.assign({}, depthRow(state.depth), I.trees[state.depth].counts) });

    const g = document.getElementById('dt-grid');
    const c1 = card(g, { title: 'Accuracy vs maximum depth', sub: 'Training, testing and 5-fold CV accuracy (band = ±1 std). Click a depth to select it.' });
    const allV = dts.flatMap((d) => [d.train, d.test, d.cvMean - d.cvStd, d.cvMean + d.cvStd]);
    chart(c1.body, (el) => lineChart(el, {
      xs: dts.map((d) => d.depth), xLabel: 'Depth ',
      series: [
        { name: 'Training', color: css('--text-3'), values: dts.map((d) => d.train), kind: 'dot' },
        { name: 'Testing', color: col, values: dts.map((d) => d.test) },
        { name: '5-fold CV', color: col, values: dts.map((d) => d.cvMean), kind: 'dash', band: dts.map((d) => [d.cvMean - d.cvStd, d.cvMean + d.cvStd]) },
      ],
      yMin: Math.floor((Math.min(...allV) - 0.01) * 20) / 20, yMax: Math.ceil((Math.max(...allV) + 0.01) * 20) / 20,
      yFmt: (v) => pct(v, 0), valFmt: (v, se, i) => (se.name === '5-fold CV' ? `${pct(v)} ± ${pct(dts[i].cvStd)}` : pct(v)),
      xTitle: 'Maximum tree depth', aria: 'Decision tree accuracy by depth', highlightX: dts.findIndex((d) => d.depth === state.depth),
      onPick: (i) => { state.depth = dts[i].depth; syncDepthSlider(); },
    }), { tags: ['depth'] });
    addDataView(c1, [{ key: 'depth', label: 'Depth' }, { key: 'train', label: 'Training', fmt: (v) => pct(v) },
      { key: 'test', label: 'Testing', fmt: (v) => pct(v) }, { key: 'cvMean', label: 'CV mean', fmt: (v) => pct(v) }, { key: 'cvStd', label: 'CV std', fmt: (v) => pct(v) }], dts);

    const c2 = card(g, { title: 'Confusion matrix', sub: `Selected depth, testing set, ${ds.test} wines.` });
    chart(c2.body, (el, d) => confusion(el, d), { tags: ['depth'], data: () => I.trees[state.depth].counts });

    const c3 = card(g, { title: 'Results for depth 5–9', sub: 'Shaded within each column (stronger blue = higher). Click a row to select that depth.', wide: true });
    const shade = (key) => {
      const vals = dts.map((d) => d[key]);
      const lo = Math.min(...vals), hi = Math.max(...vals);
      return (r) => { const c = seq(hi === lo ? 0.5 : 0.1 + 0.7 * ((r[key] - lo) / (hi - lo))); return `background:${c.bg};color:${c.ink}`; };
    };
    const metricCols = [['train', 'Train acc'], ['test', 'Test acc'], ['precision', 'Precision'], ['recall', 'Recall'], ['specificity', 'Specificity'], ['f1', 'F1'], ['cvMean', 'CV mean']];
    const rows = dts.map((d) => Object.assign({}, d, I.trees[d.depth].counts));
    const table = dataTable([
      { key: 'depth', label: 'Depth', render: (r) => (r.depth === bd ? `${r.depth}  (best CV)` : String(r.depth)) },
      ...metricCols.map(([key, lbl]) => ({ key, label: lbl, fmt: (v) => pct(v), style: shade(key) })),
      { key: 'cvStd', label: 'CV std', fmt: (v) => '± ' + pct(v) },
      { key: 'TP', label: 'TP' }, { key: 'FP', label: 'FP' }, { key: 'FN', label: 'FN' }, { key: 'TN', label: 'TN' },
      { key: 'leaves', label: 'Leaves' },
    ], rows, { rowSelected: (r) => r.depth === state.depth, rowClick: (r) => { state.depth = r.depth; syncDepthSlider(); } });
    c3.body.append(h('div', { class: 'table-wrap' }, table));
    onState(['depth'], () => table.redraw());

    const c4 = card(g, { title: 'Most important attributes', sub: 'Information gain contributed by each keyword in the tree at the selected depth (top 15). Click a keyword to highlight it.', wide: true });
    chart(c4.body, (el, d) => hBars(el, {
      rows: d.map((r) => ({ label: r.keyword, key: r.keyword, value: r.value, color: col })),
      fmt: (v) => v.toFixed(3), xMin: 0, xMax: Math.max(0.2, Math.ceil(Math.max(...Object.values(I.trees).flatMap((t) => t.importance.map((q) => q.value))) * 20) / 20),
      xTitle: 'Feature importance (share of information gain)', aria: 'Decision tree feature importance',
    }), { tags: ['depth', 'keyword'], data: () => I.trees[state.depth].importance });
  }
  function syncDepthSlider() {
    const input = document.querySelector('#dt-controls input[type="range"]');
    if (!input) return;
    input.value = D.dt.depths.findIndex((d) => d.depth === state.depth);
    input.dispatchEvent(new Event('input')); // runs the slider's handler, which updates every 'depth' view
  }

  // --------------------------------------------------------------- C) SVM
  function buildSVM() {
    const t = D.models.SVM.test;
    document.getElementById('svm-lead').textContent =
      `Linear-kernel SVM with C = 1 — the same model SVM-light trains by default (svm_learn with no options). ` +
      `It uses ${D.svm.supportVectors} support vectors and correctly labels ${t.TP + t.TN} of ${ds.test} test wines.`;
    metricKpis(document.getElementById('svm-kpis'), 'SVM');
    const g = document.getElementById('svm-grid');
    const c1 = card(g, { title: 'Confusion matrix', sub: `Testing set, ${ds.test} wines.` });
    confusion(c1.body, t);
    cvFoldCard(g, 'SVM');

    // kernel / C selector (#7)
    const K = D.svm.kernels;
    const kernels = [...new Set(K.map((r) => r.kernel))];
    const Cs = [...new Set(K.map((r) => r.C))];
    const cvOf = (kn, C) => K.find((q) => q.kernel === kn && q.C === C);
    const base = cvOf('linear', 1).cv;
    const c3 = card(g, { title: 'Kernel and C explorer', sub: 'Choose a kernel and penalty C to see their 5-fold CV accuracy. The model used above is linear, C = 1.', wide: true });
    c3.body.append(h('div', { class: 'controls in-card' },
      ctl('Kernel', segmented('SVM kernel', kernels.map((k) => ({ value: k, label: k })), state.kernel, (v) => { state.kernel = v; update('svmsel'); })),
      ctl('Penalty C', segmented('SVM penalty C', Cs.map((c) => ({ value: c, label: String(c) })), state.C, (v) => { state.C = v; update('svmsel'); }))));
    const readout = h('p', { class: 'readout', style: 'margin:0 0 10px' });
    const renderReadout = () => {
      const r = cvOf(state.kernel, state.C);
      const d = r.cv - base;
      readout.replaceChildren();
      append(readout, [`${state.kernel} kernel, C = ${state.C}: `, h('strong', { text: `${pct(r.cv)} ± ${pct(r.std)}` }), ' CV accuracy',
        state.kernel === 'linear' && state.C === 1 ? ' (the model used in this paper).'
          : [' — ', h('span', { class: d >= 0 ? 'delta-up' : 'delta-down', text: `${d >= 0 ? '+' : '−'}${Math.abs(d * 100).toFixed(2)} pts` }), ' vs linear, C = 1.']]);
    };
    renderReadout();
    onState(['svmsel'], renderReadout);
    const split = h('div', { class: 'grid', style: 'gap:24px' });
    const left = h('div'), right = h('div');
    split.append(left, right);
    c3.body.append(readout, split);
    chart(left, (el, d) => groupedBars(el, {
      groups: Cs.map((C, i) => ({ label: `C = ${C}`, values: [d[i]], onClick: () => { state.C = C; c3.body.querySelectorAll('.seg')[1].children[i].click(); } })),
      series: [{ name: 'CV accuracy', color: modelColor('SVM') }],
      yMin: 0.4, yMax: 1, yFmt: (v) => pct(v, 0), valFmt: (v) => pct(v), height: 240,
      refLine: { value: base, label: `linear, C = 1: ${pct(base)}` }, highlightGroup: Cs.indexOf(state.C),
      aria: `CV accuracy of the ${state.kernel} kernel for each C`,
    }), { tags: ['svmsel'], data: () => Cs.map((C) => cvOf(state.kernel, C).cv) });
    const vals = K.map((r) => r.cv);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const rows = kernels.map((kn) => { const r = { kernel: kn }; Cs.forEach((C) => { r['c' + C] = cvOf(kn, C).cv; }); return r; });
    const heat = dataTable([
      { key: 'kernel', label: 'Kernel' },
      ...Cs.map((C) => ({
        key: 'c' + C, label: `C = ${C}`, fmt: (v) => pct(v),
        style: (r) => { const c = seq(0.1 + 0.75 * ((r['c' + C] - lo) / (hi - lo || 1))); return `background:${c.bg};color:${c.ink}`; },
        selected: (r) => r.kernel === state.kernel && C === state.C,
      })),
    ], rows);
    right.append(h('div', { class: 'table-wrap' }, heat), h('p', { class: 'note', text: 'All kernels × C values; the outlined cell is the current selection. Small C on non-linear kernels under-fits and falls to about 50% (chance).' }));
    onState(['svmsel'], () => heat.redraw());

    divergingKeywordCard(g, {
      title: 'Most influential keywords',
      sub: 'Linear SVM weight per keyword. Click a keyword to highlight it across the page.',
      rows: D.svm.topWeights, fmt: (v) => v.toFixed(2), xTitle: 'SVM weight',
    });
  }

  // -------------------------------------------------------- D) Comparison
  const METRICS = [
    { key: 'all', label: 'All test metrics' },
    { key: 'acc', label: 'Accuracy' }, { key: 'prec', label: 'Precision' }, { key: 'rec', label: 'Recall' },
    { key: 'spec', label: 'Specificity' }, { key: 'f1', label: 'F1-score' }, { key: 'auc', label: 'ROC AUC' }, { key: 'cv', label: '5-fold CV accuracy' },
  ];
  function buildCompare() {
    const lead = document.getElementById('compare-lead');
    const renderLead = () => {
      const dt = testCounts('Decision Tree');
      lead.textContent =
        `All three models were trained on the same ${ds.train} wines and tested on the same ${ds.test}. ` +
        `Naive Bayes is the most accurate (${pct(D.models['Naive Bayes'].test.Accuracy)}) and most precise; the SVM finds the most 90+ wines. ` +
        `The decision tree at depth ${state.depth} misses ${dt.FN} of ${dt.FN + dt.TP} 90+ wines.`;
    };
    renderLead();
    onState(['depth'], renderLead);

    // model toggles (#4) + metric picker (#5)
    const toggles = h('div', { class: 'toggles', role: 'group', 'aria-label': 'Models shown' });
    MODELS.forEach((mName) => {
      const b = h('button', { type: 'button', class: 'toggle', 'aria-pressed': 'true', style: `--c:${modelColor(mName)}` }, h('span', { class: 'key' }), D.models[mName].label.replace(/ \(.*\)/, ''));
      b.addEventListener('click', () => {
        if (state.visible.has(mName)) {
          if (state.visible.size === 1) return; // keep at least one model
          state.visible.delete(mName);
        } else state.visible.add(mName);
        b.setAttribute('aria-pressed', String(state.visible.has(mName)));
        update('models');
      });
      toggles.append(b);
    });
    const select = h('select', { 'aria-label': 'Metric shown in the comparison chart' }, METRICS.map((m) => h('option', { value: m.key, text: m.label })));
    select.value = state.metric;
    select.addEventListener('change', () => { state.metric = select.value; update('metric'); });
    document.getElementById('compare-controls').append(h('div', { class: 'controls' },
      ctl('Models shown', toggles), ctl('Comparison chart shows', select)));

    const g = document.getElementById('compare-grid');
    const c0 = card(g, { title: 'Comparison table', sub: 'Testing-set metrics and 5-fold CV accuracy. Best value among the shown models in bold.', wide: true });
    const tableHolder = h('div', { class: 'table-wrap' });
    c0.body.append(tableHolder);
    const renderTable = () => {
      const rows = visibleModels().map((mName) => Object.assign({ model: mName, time: D.models[mName].time }, modelMetrics(mName), testCounts(mName)));
      const bestOf = (key, low = false) => { const vs = rows.map((r) => r[key]); const b = low ? Math.min(...vs) : Math.max(...vs); return (r) => rows.length > 1 && r[key] === b; };
      tableHolder.replaceChildren(dataTable([
        { key: 'model', label: 'Model', render: (r) => h('span', { class: 'model-name' }, h('span', { class: 'key', style: `background:${modelColor(r.model)}` }), label(r.model)) },
        { key: 'acc', label: 'Accuracy', fmt: (v) => pct(v), best: bestOf('acc') },
        { key: 'prec', label: 'Precision', fmt: (v) => pct(v), best: bestOf('prec') },
        { key: 'rec', label: 'Recall', fmt: (v) => pct(v), best: bestOf('rec') },
        { key: 'spec', label: 'Specificity', fmt: (v) => pct(v), best: bestOf('spec') },
        { key: 'f1', label: 'F1', fmt: (v) => pct(v), best: bestOf('f1') },
        { key: 'auc', label: 'AUC', fmt: (v) => v.toFixed(3), best: bestOf('auc') },
        { key: 'cv', label: '5-fold CV', fmt: (v, r) => `${pct(v)} ± ${pct(r.cvs)}`, best: bestOf('cv') },
        { key: 'TP', label: 'TP' }, { key: 'FP', label: 'FP' }, { key: 'FN', label: 'FN' }, { key: 'TN', label: 'TN' },
      ], rows));
    };
    renderTable();
    onState(['models', 'depth'], renderTable);
    c0.body.append(h('p', { class: 'note', text: 'AUC for the decision tree is for the best-CV depth; all other decision-tree values follow the depth slider.' }));

    const c1 = card(g, { title: 'Metrics side by side', sub: 'Pick a single metric above to rank the shown models by it.', wide: true });
    const metricsAll = [['acc', 'Accuracy'], ['prec', 'Precision'], ['rec', 'Recall'], ['spec', 'Specificity'], ['f1', 'F1-score']];
    chart(c1.body, (el, d) => {
      if (d.mode === 'all') {
        groupedBars(el, {
          groups: metricsAll.map(([, lbl], i) => ({ label: lbl, values: d.values[i] })),
          series: d.models.map((mName) => ({ name: label(mName), color: modelColor(mName) })),
          yMin: 0.5, yMax: 1, yFmt: (v) => pct(v, 0), valFmt: (v) => pct(v), height: 300, aria: 'Metric comparison of the shown models',
        });
      } else {
        const isAuc = d.mode === 'auc';
        hBars(el, {
          rows: d.rows.map((r) => ({ label: label(r.model), value: r.value, color: modelColor(r.model) })),
          fmt: isAuc ? (v) => v.toFixed(3) : (v) => pct(v), tickFmt: isAuc ? (v) => v.toFixed(1) : (v) => pct(v, 0), xMin: 0, xMax: 1, rowH: 40,
          xTitle: METRICS.find((m) => m.key === d.mode).label, aria: 'Models ranked by metric',
        });
      }
    }, {
      tags: ['models', 'metric', 'depth'],
      data: () => {
        const ms = visibleModels();
        if (state.metric === 'all') return { mode: 'all', models: ms, values: metricsAll.map(([key]) => ms.map((mName) => modelMetrics(mName)[key])) };
        return { mode: state.metric, rows: ms.map((mName) => ({ model: mName, value: modelMetrics(mName)[state.metric] })).sort((a, b) => b.value - a.value) };
      },
    });

    const c2 = card(g, { title: 'ROC curves', sub: 'Trade-off between finding 90+ wines and false alarms across all thresholds.' });
    chart(c2.body, (el) => rocChart(el, { models: visibleModels() }), { tags: ['models'] });

    const c3 = card(g, { title: 'Cross-validation stability', sub: 'Each dot is one of the 5 folds; the bar is the mean (decision tree at its best-CV depth).' });
    chart(c3.body, (el) => cvStrip(el, visibleModels()), { tags: ['models'] });

    const c4 = card(g, { title: 'Confusion matrices', sub: `Testing set, ${ds.test} wines (${ds.testPos} rated 90+, ${ds.testNeg} rated 90-).`, wide: true });
    chart(c4.body, (el, d) => {
      const row = h('div', { class: 'cm-row', style: `--n:${d.length}` });
      d.forEach((r) => { const cell = h('div'); confusion(cell, r.c, label(r.model)); row.append(cell); });
      el.append(row);
    }, { tags: ['models', 'depth'], data: () => visibleModels().map((mName) => ({ model: mName, c: testCounts(mName) })) });
  }

  // ------------------------------------------------- threshold explorer
  function buildThreshold() {
    document.getElementById('threshold-lead').textContent =
      'Naive Bayes predicts 90+ when P(90+) is above 50%; the SVM when its decision score is above 0. ' +
      'Moving that threshold trades false alarms (FP) against missed 90+ wines (FN). The default thresholds give the results reported above.';
    const svmVals = I.test.map((w) => w.svm);
    const svmLo = Math.floor(Math.min(...svmVals) * 10) / 10, svmHi = Math.ceil(Math.max(...svmVals) * 10) / 10;
    const DEF = { 'Naive Bayes': 0.5, SVM: 0 };

    const sliderHolder = h('div', { class: 'ctl grow' });
    const renderSlider = () => {
      const isNB = state.thrModel === 'Naive Bayes';
      const v = state.thr[state.thrModel];
      const input = h('input', {
        type: 'range', min: isNB ? 0.01 : svmLo, max: isNB ? 0.99 : svmHi, step: isNB ? 0.01 : 0.05, value: v,
        'aria-label': isNB ? 'Naive Bayes probability threshold' : 'SVM decision score threshold',
      });
      const val = h('span', { class: 'ctl-value', text: isNB ? pct(v, 0) : v.toFixed(2) });
      input.addEventListener('input', () => {
        state.thr[state.thrModel] = +input.value;
        val.textContent = isNB ? pct(+input.value, 0) : (+input.value).toFixed(2);
        update('thr');
      });
      sliderHolder.replaceChildren(
        h('span', { class: 'ctl-label' }, isNB ? 'Predict 90+ when P(90+) is above ' : 'Predict 90+ when SVM score is above ', val),
        input,
        h('div', { class: 'range-ticks' }, h('span', { text: isNB ? '1%' : svmLo.toFixed(1) }), h('span', { text: isNB ? 'default 50%' : 'default 0' }), h('span', { text: isNB ? '99%' : svmHi.toFixed(1) })));
    };
    renderSlider();
    const reset = h('button', { class: 'btn', type: 'button', text: 'Reset to default' });
    reset.addEventListener('click', () => { state.thr[state.thrModel] = DEF[state.thrModel]; renderSlider(); update('thr'); });
    document.getElementById('threshold-controls').append(h('div', { class: 'controls' },
      ctl('Model', segmented('Threshold model', [{ value: 'Naive Bayes', label: 'Naive Bayes' }, { value: 'SVM', label: 'SVM' }], state.thrModel, (v) => { state.thrModel = v; renderSlider(); update('thr'); })),
      sliderHolder, reset));

    const cur = () => thresholdCounts(state.thrModel, state.thr[state.thrModel]);
    chart(document.getElementById('threshold-kpis'), (el, d) => {
      const sc = scores(d);
      const def = scores(thresholdCounts(state.thrModel, DEF[state.thrModel]));
      const delta = (a, b) => { const x = (a - b) * 100; return Math.abs(x) < 0.005 ? 'same as default' : `${x > 0 ? '+' : '−'}${Math.abs(x).toFixed(2)} pts vs default`; };
      kpi(el, { label: 'Accuracy', value: pct(sc.acc), sub: delta(sc.acc, def.acc), color: modelColor(state.thrModel) });
      kpi(el, { label: 'Precision', value: pct(sc.prec), sub: delta(sc.prec, def.prec) });
      kpi(el, { label: 'Recall (sensitivity)', value: pct(sc.rec), sub: delta(sc.rec, def.rec) });
      kpi(el, { label: 'Specificity', value: pct(sc.spec), sub: delta(sc.spec, def.spec) });
      kpi(el, { label: 'F1-score', value: pct(sc.f1), sub: delta(sc.f1, def.f1) });
      kpi(el, { label: 'Predicted 90+', value: String(Math.round(d.TP + d.FP)), sub: `of ${ds.test} test wines` });
    }, { tags: ['thr'], data: cur });

    const g = document.getElementById('threshold-grid');
    const c1 = card(g, { title: 'Confusion matrix at this threshold', sub: `Testing set, ${ds.test} wines.` });
    chart(c1.body, (el, d) => confusion(el, d), { tags: ['thr'], data: cur });

    const c2 = card(g, { title: 'Where the threshold cuts', sub: 'Model scores of the test wines by real grade. Wines right of the line are predicted 90+.' });
    chart(c2.body, (el, d) => scoreHistogram(el, { model: state.thrModel, thr: d.thr }), { tags: ['thr'], data: () => ({ thr: state.thr[state.thrModel] }) });

    const c3 = card(g, { title: 'Position on the ROC curve', sub: 'The dot is the current threshold; dragging the slider moves it along the curve.' });
    chart(c3.body, (el, d) => rocChart(el, {
      models: [state.thrModel],
      point: { fpr: d.FP / (d.FP + d.TN), tpr: d.TP / (d.TP + d.FN), color: modelColor(state.thrModel), label: state.thrModel === 'SVM' ? state.thr.SVM.toFixed(2) : pct(state.thr['Naive Bayes'], 0) },
    }), { tags: ['thr'], data: cur });

    const c4 = card(g, { title: 'Metrics across thresholds', sub: 'Accuracy, precision and recall for every threshold; the shaded column is the current one. Click to jump to a threshold.' });
    chart(c4.body, (el) => {
      const isNB = state.thrModel === 'Naive Bayes';
      const xs = [];
      if (isNB) for (let v = 0.05; v < 0.951; v += 0.05) xs.push(+v.toFixed(2));
      else for (let v = Math.ceil(svmLo * 2) / 2; v <= svmHi + 1e-9; v += 0.5) xs.push(+v.toFixed(2));
      const sc = xs.map((t) => scores(thresholdCounts(state.thrModel, t)));
      const curT = state.thr[state.thrModel];
      let hi = 0;
      xs.forEach((t, i) => { if (Math.abs(t - curT) < Math.abs(xs[hi] - curT)) hi = i; });
      const col = modelColor(state.thrModel);
      lineChart(el, {
        xs: xs.map((t) => (isNB ? Math.round(t * 100) + '%' : t)), xLabel: 'Threshold ',
        series: [
          { name: 'Accuracy', color: col, values: sc.map((q) => q.acc) },
          { name: 'Precision', color: css('--text-2'), values: sc.map((q) => q.prec), kind: 'dash' },
          { name: 'Recall', color: css('--text-3'), values: sc.map((q) => q.rec), kind: 'dot' },
        ],
        yMin: 0, yMax: 1, yFmt: (v) => pct(v, 0), valFmt: (v) => pct(v), markers: false, height: 300,
        xTitle: isNB ? 'Naive Bayes threshold on P(90+)' : 'SVM threshold on the decision score', aria: 'Metrics across thresholds',
        highlightX: hi,
        onPick: (i) => {
          state.thr[state.thrModel] = xs[i];
          renderSlider();
          update('thr');
        },
      });
    }, { tags: ['thr'] });
  }

  // -------------------------------------------------- E) Knowledge gained
  function topLists(minCount) {
    const elig = D.keywords.all.filter((r) => r.all >= minCount && r.p90 != null);
    const pos = elig.slice().sort((a, b) => b.p90 - a.p90 || b.all - a.all).slice(0, 20);
    const neg = elig.slice().sort((a, b) => a.p90 - b.p90 || b.all - a.all).slice(0, 20);
    return { pos, neg, n: elig.length };
  }
  function buildKnowledge() {
    const K = D.keywords;
    const lead = document.getElementById('knowledge-lead');
    const renderLead = () => {
      lead.textContent =
        `What the reviews reveal about 90+ wines. "Common" keywords are frequent in all reviews; "90+ keywords" appear in at least ${state.minCount} reviews ` +
        `and are mostly found in wines rated 90 or higher. Click any keyword to highlight it in every chart.`;
    };
    renderLead();
    onState(['minCount'], renderLead);

    // minimum-review filter (#9)
    const MINS = [5, 10, 15, 20, 30, 50];
    const countNote = h('span', { class: 'note', style: 'margin:0' });
    const renderCount = () => { countNote.textContent = `${topLists(state.minCount).n} of ${K.all.length} keywords qualify.`; };
    renderCount();
    onState(['minCount'], renderCount);
    document.getElementById('knowledge-controls').append(h('div', { class: 'controls' },
      h('div', { class: 'ctl grow' }, h('span', { class: 'ctl-label' }, 'Minimum reviews for a keyword to be ranked (90+/90- lists and scatter)'),
        stepSlider({ values: MINS, value: state.minCount, aria: 'Minimum reviews per keyword', onInput: (v) => { state.minCount = v; update('minCount'); } })),
      countNote));

    const g = document.getElementById('knowledge-grid');
    const pos = css('--pos'), neg = css('--neg');
    const kwCols = [
      { key: 'keyword', label: 'Keyword' }, { key: 'all', label: 'Reviews' }, { key: 'pos', label: 'in 90+' }, { key: 'neg', label: 'in 90-' },
      { key: 'p90', label: 'P(90+ | keyword)', fmt: (v) => pct1(v) },
    ];

    const c1 = card(g, { title: 'Top 20 most common keywords', sub: 'Number of reviews containing each keyword, split by class.' });
    chart(c1.body, (el) => stackedHBars(el, {
      rows: K.common.map((r) => ({ label: r.keyword, key: r.keyword, parts: [r.pos, r.neg] })),
      series: [{ name: '90+ wines', color: pos }, { name: '90- wines', color: neg }],
      fmt: (v) => String(Math.round(v)), xTitle: 'Reviews', aria: 'Most common keywords',
    }), { tags: ['keyword'] });
    addDataView(c1, kwCols, K.common);

    const c2 = card(g, { title: 'Top 20 keywords of 90+ wines', sub: 'Share of reviews with the keyword that scored 90+.' });
    chart(c2.body, (el, d) => hBars(el, {
      rows: d.map((r) => ({ label: r.keyword, key: r.keyword, value: r.p90, color: pos, tip: [{ value: pct1(r.p90), label: 'are 90+', color: pos }, { value: String(Math.round(r.all)), label: 'reviews' }] })),
      fmt: (v) => pct(v, 0), xMin: 0, xMax: 1, xTitle: '% of wines with the keyword rated 90+', aria: '90+ keywords',
    }), { tags: ['minCount', 'keyword'], data: () => topLists(state.minCount).pos });
    const fill2 = addDataView(c2, kwCols, () => topLists(state.minCount).pos);
    onState(['minCount'], fill2);

    const c3 = card(g, { title: 'Top 20 keywords of 90- wines', sub: 'Keywords whose wines rarely reach 90.' });
    chart(c3.body, (el, d) => hBars(el, {
      rows: d.map((r) => ({ label: r.keyword, key: r.keyword, value: 1 - r.p90, color: neg, tip: [{ value: pct1(1 - r.p90), label: 'are 90-', color: neg }, { value: String(Math.round(r.all)), label: 'reviews' }] })),
      fmt: (v) => pct(v, 0), xMin: 0, xMax: 1, xTitle: '% of wines with the keyword rated below 90', aria: '90- keywords',
    }), { tags: ['minCount', 'keyword'], data: () => topLists(state.minCount).neg });
    const fill3 = addDataView(c3, kwCols, () => topLists(state.minCount).neg);
    onState(['minCount'], fill3);

    const c4 = card(g, { title: 'Keywords the models agree on', sub: 'Top-20 indicators of 90+ shared by the Naive Bayes ratio and the SVM weights. Click to highlight.' });
    const strong = new Set(K.agreedAll);
    const chips = h('div', { class: 'chips' });
    const renderChips = () => {
      chips.replaceChildren(...K.agreed.map((w) => {
        const chip = h('span', { class: 'chip' + (strong.has(w) ? ' strong' : '') + (state.keyword === w ? ' sel' : ''), tabindex: '0', role: 'button', text: w });
        bindClick(chip, () => selectKeyword(w));
        return chip;
      }));
    };
    renderChips();
    onState(['keyword'], renderChips);
    c4.body.append(chips,
      h('p', { class: 'note', text: `Outlined: also among the decision tree's 20 most important attributes (${K.agreedAll.join(', ')}).` }),
      h('p', { class: 'note', text: 'Descriptors of structure and length (LONG, LONG FINISH, CONCENTRATED, FULL-BODIED) and praise words (BEAUTY, ELEGANT, FINESSE) mark 90+ wines, while LIGHT-BODIED, FRESH and simple fruit notes dominate 90- reviews.' }));

    const c5 = card(g, { title: 'Keyword frequency: 90+ vs 90- wines', sub: 'Each dot is a keyword that passes the minimum-review filter. Above the dashed line = more common in 90+ reviews. Click a dot to select it.', wide: true });
    c5.el.id = 'kw-scatter';
    chart(c5.body, (el) => keywordScatter(el, state.minCount), { tags: ['minCount', 'keyword'] });

    const c6 = card(g, { title: 'Keyword explorer', sub: `All ${K.all.length} attributes. Sort by a column, type to filter, click a row to highlight the keyword everywhere.`, wide: true });
    const input = h('input', { type: 'search', placeholder: 'Search keywords…', 'aria-label': 'Search keywords' });
    const count = h('span', { class: 'count' });
    const table = dataTable([
      { key: 'keyword', label: 'Keyword' }, { key: 'all', label: 'Reviews' }, { key: 'pos', label: 'in 90+' }, { key: 'neg', label: 'in 90-' },
      { key: 'pctPos', label: '% of 90+', fmt: (v) => v.toFixed(1) + '%' }, { key: 'pctNeg', label: '% of 90-', fmt: (v) => v.toFixed(1) + '%' },
      { key: 'p90', label: 'P(90+ | kw)', fmt: (v) => pct1(v) },
      { key: 'nb', label: 'NB log-ratio', fmt: (v) => v.toFixed(2) }, { key: 'svm', label: 'SVM weight', fmt: (v) => v.toFixed(2) },
    ], K.all, { sortable: true, rowClick: (r) => selectKeyword(r.keyword), rowSelected: (r) => r.keyword === state.keyword });
    const filter = () => {
      const q = input.value.trim().toUpperCase();
      const r = q ? K.all.filter((x) => x.keyword.includes(q)) : K.all;
      table.setRows(r);
      count.textContent = `${r.length} of ${K.all.length} keywords`;
    };
    input.addEventListener('input', filter);
    onState(['keyword'], () => table.redraw());
    c6.body.append(h('div', { class: 'toolbar' }, input, count), h('div', { class: 'table-wrap tall' }, table));
    filter();
  }

  // --------------------------------------------------- predict a wine
  const escapeRe = (str) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  const KW_RE = FEATS.map((f, j) => ({ j, re: new RegExp('(^|[^A-Z])' + escapeRe(f).replace(/[- ]/g, '[- ]?') + '(S|ES)?(?![A-Z])') }));
  function extractKeywords(text) {
    const T = text.toUpperCase();
    const out = new Set();
    KW_RE.forEach((k) => { if (k.re.test(T)) out.add(k.j); });
    return out;
  }
  function nbPredict(set) {
    let l0 = I.nb.logPrior[0], l1 = I.nb.logPrior[1];
    I.nb.p1.forEach((p, j) => {
      if (set.has(j)) { l0 += Math.log(p[0]); l1 += Math.log(p[1]); }
      else { l0 += Math.log(1 - p[0]); l1 += Math.log(1 - p[1]); }
    });
    const mx = Math.max(l0, l1);
    const e0 = Math.exp(l0 - mx), e1 = Math.exp(l1 - mx);
    return e1 / (e0 + e1);
  }
  const svmPredict = (set) => { let sc = I.svm.intercept; set.forEach((j) => { sc += I.svm.weights[j]; }); return sc; };
  function dtPredict(set, depth) {
    const t = I.trees[depth];
    let n = 0;
    const path = [];
    while (t.left[n] !== -1) {
      const f = t.feature[n];
      const has = set.has(f);
      path.push({ keyword: FEATS[f], present: has });
      n = has ? t.right[n] : t.left[n];
    }
    return { p: t.value[n][1], pred: t.value[n][1] > t.value[n][0] ? 1 : 0, samples: t.samples[n], path };
  }
  /* evidence a present keyword adds to the Naive Bayes log-odds of 90+ (vs. the keyword being absent) */
  const nbEvidence = (j) => {
    const [p0, p1] = I.nb.p1[j];
    return Math.log(p1 / p0) - Math.log((1 - p1) / (1 - p0));
  };

  function buildPredict() {
    const P = state.predict;
    const lastVerdict = {};
    const flipAt = {};
    let prevChips = new Set();
    document.getElementById('predict-lead').textContent =
      'Load a wine from the test set (the models never saw these during training), paste a tasting note, or add keywords by hand. ' +
      'All three models re-score the wine instantly using the exact parameters learned in Python.';
    const top = document.getElementById('predict-top');
    const inCard = card(top, { title: 'Wine', sub: 'Choose an input; the keyword list below is what the models see.' });
    const outCard = card(top, { title: 'Predictions', sub: ' ' });

    // default: first 90+ test wine that all three models get right
    if (P.wine == null && !P.kw.size) {
      const idx = I.test.findIndex((w, i) => w.label === 1 && w.nb > 0.5 && w.svm > 0 && I.trees[state.depth].pred[i] === 1 && w.kw.length >= 6);
      P.wine = idx >= 0 ? idx : 0;
      P.kw = new Set(I.test[P.wine].kw);
    }

    const sel = h('select', { 'aria-label': 'Load a wine from the test set' },
      h('option', { value: '', text: '— choose a test wine —' }),
      h('optgroup', { label: 'Rated 90+' }, I.test.map((w, i) => (w.label ? h('option', { value: i, text: w.name }) : null))),
      h('optgroup', { label: 'Rated below 90' }, I.test.map((w, i) => (!w.label ? h('option', { value: i, text: w.name }) : null))));
    sel.value = P.wine ?? '';
    sel.addEventListener('change', () => {
      if (sel.value === '') return;
      P.wine = +sel.value; P.kw = new Set(I.test[P.wine].kw); P.modified = false; P.text = ''; ta.value = '';
      update('predict');
    });
    const randomBtn = h('button', { class: 'btn', type: 'button', text: 'Random test wine' });
    randomBtn.addEventListener('click', () => { sel.value = String(Math.floor(Math.random() * I.test.length)); sel.dispatchEvent(new Event('change')); });

    const ta = h('textarea', { placeholder: 'e.g. Rich and full-bodied, with a long finish of dark plum and spice…', 'aria-label': 'Tasting note' });
    let tTimer;
    ta.addEventListener('input', () => {
      clearTimeout(tTimer);
      tTimer = setTimeout(() => {
        P.text = ta.value;
        P.kw = extractKeywords(ta.value);
        P.wine = null; P.modified = false; sel.value = '';
        update('predict');
      }, 200);
    });

    const dl = h('datalist', { id: 'kw-list' }, FEATS.map((f) => h('option', { value: f })));
    const kwInput = h('input', { class: 'kw-input', list: 'kw-list', placeholder: 'Add a keyword (e.g. LONG FINISH)', 'aria-label': 'Add a keyword' });
    const addBtn = h('button', { class: 'btn', type: 'button', text: 'Add' });
    const addKw = () => {
      const j = FEATS.indexOf(kwInput.value.trim().toUpperCase());
      if (j < 0) { kwInput.setCustomValidity('Not one of the 305 wheel attributes'); kwInput.reportValidity(); return; }
      kwInput.setCustomValidity('');
      P.kw.add(j); P.modified = P.wine != null; kwInput.value = '';
      update('predict');
    };
    addBtn.addEventListener('click', addKw);
    kwInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); addKw(); } });
    kwInput.addEventListener('input', () => kwInput.setCustomValidity(''));
    const clearBtn = h('button', { class: 'btn', type: 'button', text: 'Clear all' });
    clearBtn.addEventListener('click', () => { P.kw = new Set(); P.wine = null; P.modified = false; sel.value = ''; ta.value = ''; update('predict'); });

    const chipBox = h('div');
    const renderChips = () => {
      const list = [...P.kw].sort((a, b) => FEATS[a].localeCompare(FEATS[b]));
      chipBox.replaceChildren(list.length
        ? h('div', { class: 'chips' }, list.map((j) => {
          const rm = h('button', { type: 'button', 'aria-label': `Remove ${FEATS[j]}`, text: '×' });
          rm.addEventListener('click', () => { P.kw.delete(j); P.modified = P.wine != null; update('predict'); });
          return h('span', { class: 'chip' + (prevChips.size && !prevChips.has(j) ? ' pop-chip' : '') }, FEATS[j], rm);
        }))
        : h('p', { class: 'empty', text: 'No keywords yet — every model then predicts from its baseline alone.' }));
      prevChips = new Set(list.length ? list : [-1]);
    };
    renderChips();
    onState(['predict'], renderChips);

    inCard.body.append(
      h('div', { class: 'field' }, h('span', { class: 'ctl-label', text: 'Load a wine from the test set' }), h('div', { class: 'kw-add' }, sel, randomBtn)),
      h('div', { class: 'field' }, h('span', { class: 'ctl-label', text: '…or paste a tasting note (keywords are matched against the 305 wheel attributes)' }), ta),
      h('div', { class: 'field' }, h('span', { class: 'ctl-label', text: '…or add keywords' }), h('div', { class: 'kw-add' }, kwInput, addBtn, clearBtn), dl),
      h('div', { class: 'field' }, h('span', { class: 'ctl-label' }, 'Keywords the models see'), chipBox));

    // predictions (animated values)
    const head = outCard.head.querySelector('.sub');
    const renderHead = () => {
      head.textContent = P.wine != null
        ? `${I.test[P.wine].name}${P.modified ? ' (keywords edited)' : ''}`
        : (P.text ? 'Your tasting note' : 'Hand-picked keywords');
    };
    renderHead();
    onState(['predict', 'depth'], renderHead);

    chart(outCard.body, (el, d) => {
      const wine = P.wine != null && !P.modified ? I.test[P.wine] : null;
      if (wine) {
        el.append(h('div', { class: 'truth' }, 'Real grade:', h('span', { class: 'pill ' + (wine.label ? 'pos' : 'neg'), text: wine.label ? '90+' : 'below 90' })));
      }
      const verdict = (yes, key) => {
        // pulse when a model's verdict flips; the pill is rebuilt every animation frame, so the
        // animation is resumed at its elapsed time instead of restarting
        const now = performance.now();
        if (lastVerdict[key] !== undefined && lastVerdict[key] !== yes) flipAt[key] = now;
        lastVerdict[key] = yes;
        const age = now - (flipAt[key] ?? -1e9);
        return h('span', {
          class: 'pill ' + (yes ? 'pos' : 'neg') + (age < 600 ? ' flip' : ''),
          style: age < 600 ? `animation-delay:-${Math.round(age)}ms` : null,
          text: yes ? 'Predicts 90+' : 'Predicts below 90',
        });
      };
      const right = (pred) => (wine ? h('span', { class: 'ok', text: pred === wine.label ? ' ✓ correct' : ' ✗ wrong' }) : null);
      const meter = (value, lo, hi, mid, color, fmt, ariaLbl) => {
        const W = Math.max(el.clientWidth - 30, 200);
        const svg = svgRoot(W, 34, ariaLbl);
        const x = lin(lo, hi, 4, W - 4);
        svg.append(s('rect', { x: 4, y: 8, width: W - 8, height: 10, rx: 5, fill: css('--surface-2') }));
        const x0 = x(mid), x1 = x(clamp(value, lo, hi));
        svg.append(s('path', { d: hBarPath(x0, x1, 8, 10, 4), fill: color }));
        svg.append(s('line', { x1: x0, x2: x0, y1: 3, y2: 23, stroke: css('--text-2'), 'stroke-width': 1.5 }));
        svg.append(s('text', { class: 'tick', x: x0, y: 32, 'text-anchor': 'middle', text: fmt(mid) }));
        svg.append(s('text', { class: 'tick', x: 4, y: 32, text: fmt(lo) }));
        svg.append(s('text', { class: 'tick', x: W - 4, y: 32, 'text-anchor': 'end', text: fmt(hi) }));
        return svg;
      };
      const rows = h('div', { class: 'verdicts' });
      // Naive Bayes
      const nbYes = d.nb > 0.5;
      rows.append(h('div', { class: 'verdict-row' },
        h('div', { class: 'verdict-top' }, h('span', { class: 'verdict-model' }, h('span', { class: 'key', style: `background:${modelColor('Naive Bayes')}` }), 'Naive Bayes'), verdict(nbYes, 'nb')),
        meter(d.nb, 0, 1, 0.5, modelColor('Naive Bayes'), (v) => pct(v, 0), 'Naive Bayes probability of 90+'),
        h('div', { class: 'verdict-sub' }, 'P(90+) = ', h('strong', { text: pct1(d.nb) }), right(nbYes ? 1 : 0))));
      // SVM
      const svmYes = d.svm > 0;
      rows.append(h('div', { class: 'verdict-row' },
        h('div', { class: 'verdict-top' }, h('span', { class: 'verdict-model' }, h('span', { class: 'key', style: `background:${modelColor('SVM')}` }), 'SVM (linear)'), verdict(svmYes, 'svm')),
        meter(d.svm, -4, 4, 0, modelColor('SVM'), (v) => (v > 0 ? '+' : '') + v, 'SVM decision score'),
        h('div', { class: 'verdict-sub' }, 'Decision score w·x + b = ', h('strong', { text: d.svm.toFixed(2) }), right(svmYes ? 1 : 0))));
      // Decision tree
      const dt = dtPredict(P.kw, state.depth);
      const pathList = h('ol', { class: 'path' }, dt.path.map((st) => h('li', {}, h('b', { text: st.keyword }), st.present ? ' present → right' : ' absent → left')));
      rows.append(h('div', { class: 'verdict-row' },
        h('div', { class: 'verdict-top' }, h('span', { class: 'verdict-model' }, h('span', { class: 'key', style: `background:${modelColor('Decision Tree')}` }), `Decision Tree (depth ${state.depth})`), verdict(dt.pred === 1, 'dt')),
        meter(d.dt, 0, 1, 0.5, modelColor('Decision Tree'), (v) => pct(v, 0), 'Share of 90+ training wines in the leaf'),
        h('div', { class: 'verdict-sub' }, `Leaf: ${pct1(dt.p)} of ${dt.samples} training wines were 90+`, right(dt.pred)),
        h('details', {}, h('summary', { class: 'note', style: 'cursor:pointer', text: `Path through the tree (${dt.path.length} questions)` }), pathList)));
      el.append(rows);
    }, {
      tags: ['predict', 'depth'],
      data: () => ({ nb: nbPredict(P.kw), svm: svmPredict(P.kw), dt: dtPredict(P.kw, state.depth).p }),
    });

    // contributions
    const g = document.getElementById('predict-grid');
    const c1 = card(g, { title: 'Why: Naive Bayes evidence', sub: 'How much each keyword moves the log-odds of 90+ compared with the keyword being absent.' });
    chart(c1.body, (el, d) => hBars(el, {
      rows: d.map((r) => ({ label: r.keyword, key: r.keyword, value: r.value, color: r.value >= 0 ? css('--pos') : css('--neg') })),
      fmt: (v) => (v > 0 ? '+' : '') + v.toFixed(2), xTitle: 'Change in log-odds of 90+', emptyText: 'Add keywords to see their evidence.',
    }), { tags: ['predict', 'keyword'], data: () => [...P.kw].map((j) => ({ keyword: FEATS[j], value: nbEvidence(j) })).sort((a, b) => b.value - a.value) });

    const c2 = card(g, { title: 'Why: SVM weights', sub: `Each present keyword adds its weight to the score; the intercept b = ${I.svm.intercept.toFixed(2)}.` });
    chart(c2.body, (el, d) => hBars(el, {
      rows: d.map((r) => ({ label: r.keyword, key: r.keyword, value: r.value, color: r.value >= 0 ? css('--pos') : css('--neg') })),
      fmt: (v) => (v > 0 ? '+' : '') + v.toFixed(2), xTitle: 'SVM weight', emptyText: 'Add keywords to see their weights.',
    }), { tags: ['predict', 'keyword'], data: () => [...P.kw].map((j) => ({ keyword: FEATS[j], value: I.svm.weights[j] })).sort((a, b) => b.value - a.value) });
  }


  // ------------------------------------------------------------- motion
  /* Count a KPI value up from zero (only plain numbers such as 87.13%, 1,010, 0.944). */
  function countUp(container, delay) {
    container.querySelectorAll('.kpi .value').forEach((v) => {
      const txt = v.textContent;
      const m = txt.match(/^([\d,]+(?:\.\d+)?)(%?)$/);
      if (!m) return;
      const target = parseFloat(m[1].replace(/,/g, ''));
      const dec = (m[1].split('.')[1] || '').length;
      const comma = m[1].includes(',');
      const fmt = (n) => (comma ? n.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec }) : n.toFixed(dec)) + m[2];
      const t0 = performance.now() + delay;
      const dur = 900;
      v.textContent = fmt(0);
      const step = (now) => {
        if (!v.isConnected || v.dataset.done) return;
        const t = clamp((now - t0) / dur, 0, 1);
        v.textContent = t < 1 ? fmt(target * (1 - (1 - t) ** 3)) : txt;
        if (t < 1) requestAnimationFrame(step);
      };
      requestAnimationFrame(step);
      setTimeout(() => { if (v.isConnected) { v.dataset.done = '1'; v.textContent = txt; } }, delay + dur + 200);
    });
  }

  function setupMotion() {
    if (reduceMotion) return;
    const head = document.querySelector('header.top .head-row > div');
    head.classList.add('hero-in');
    document.querySelector('.head-actions').classList.add('hero-in-btns');
    if (!('IntersectionObserver' in window)) return;
    const els = [...document.querySelectorAll('main section > h2, main section > .lead, main .findings, main .kpis, main .controls:not(.in-card), main .card')];
    els.forEach((el) => el.classList.add('anim-pre'));
    const reveal = (el, delay) => {
      el.style.setProperty('--d', delay + 'ms');
      el.classList.remove('anim-pre');
      el.classList.add('anim-in', 'enter');
      if (el.classList.contains('kpis')) countUp(el, delay + 150);
      setTimeout(() => { el.classList.remove('anim-in', 'enter'); el.style.removeProperty('--d'); }, delay + 2400);
    };
    const io = new IntersectionObserver((entries) => {
      const shown = entries.filter((e) => e.isIntersecting).map((e) => e.target);
      shown.sort((a, b) => {
        const ra = a.getBoundingClientRect(), rb = b.getBoundingClientRect();
        return ra.top - rb.top || ra.left - rb.left;
      });
      shown.forEach((el, i) => { io.unobserve(el); reveal(el, Math.min(i, 6) * 70); });
    }, { rootMargin: '0px 0px -6% 0px', threshold: 0.06 });
    els.forEach((el) => io.observe(el));
    // safety net: anything on screen that is still hidden gets revealed (e.g. if the observer
    // never fired because the page opened in a background tab)
    const sweep = () => {
      if (document.hidden) return;
      let n = 0;
      els.forEach((el) => {
        if (!el.classList.contains('anim-pre')) return;
        const r = el.getBoundingClientRect();
        if (r.top < innerHeight && r.bottom > 0) { io.unobserve(el); reveal(el, Math.min(n++, 6) * 70); }
      });
    };
    setTimeout(sweep, 1200);
    document.addEventListener('visibilitychange', () => setTimeout(sweep, 400));
    let st;
    window.addEventListener('scroll', () => { clearTimeout(st); st = setTimeout(sweep, 600); }, { passive: true });
    window.addEventListener('beforeprint', () => els.forEach((el) => { io.unobserve(el); el.classList.remove('anim-pre'); }));
  }


  // ------------------------------------------------------- decoration
  function setupDecor() {
    // result badges in the header (colors follow the theme through CSS variables)
    const stats = h('div', { class: 'hero-stats' },
      MODELS.map((m, i) => h('span', { class: 'hero-stat', style: `--i:${i};--c:var(${MODEL_VAR[m]})` },
        h('span', { class: 'key' }), D.models[m].label.replace(/ \(.*\)/, ''), h('strong', { text: pct(D.models[m].test.Accuracy) }))),
      h('span', { class: 'hero-stat', style: '--i:3;--c:var(--amber)' }, h('span', { class: 'key' }), h('strong', { text: ds.wines.toLocaleString() }), 'wines · ', h('strong', { text: String(ds.attributes) }), 'keywords'));
    document.querySelector('header.top .head-row > div').append(stats);

    // scroll progress bar + back-to-top button
    const bar = document.querySelector('.progress');
    const toTop = h('button', { class: 'to-top', type: 'button', 'aria-label': 'Back to top', text: '↑' });
    toTop.addEventListener('click', () => window.scrollTo({ top: 0, behavior: reduceMotion ? 'auto' : 'smooth' }));
    document.body.append(toTop);
    const onScroll = () => {
      const max = document.documentElement.scrollHeight - innerHeight;
      if (bar) bar.style.transform = `scaleX(${max > 0 ? clamp(scrollY / max, 0, 1) : 0})`;
      toTop.classList.toggle('show', scrollY > innerHeight * 0.8);
    };
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    onScroll();

    // ripple on button presses
    if (reduceMotion) return;
    document.addEventListener('pointerdown', (e) => {
      const b = e.target.closest('.btn, .seg button, .toggle, button.theme');
      if (!b) return;
      const r = b.getBoundingClientRect();
      const size = Math.max(r.width, r.height);
      const dot = h('span', { class: 'ripple', style: `width:${size}px;height:${size}px;left:${e.clientX - r.left - size / 2}px;top:${e.clientY - r.top - size / 2}px` });
      b.append(dot);
      setTimeout(() => dot.remove(), 600);
    });
  }

  // ------------------------------------------------------------- startup
  const CONTAINERS = ['whatsnew', 'findings', 'overview-kpis', 'overview-grid', 'nb-kpis', 'nb-grid', 'dt-controls', 'dt-kpis', 'dt-grid',
    'svm-kpis', 'svm-grid', 'compare-controls', 'compare-grid', 'threshold-controls', 'threshold-kpis', 'threshold-grid',
    'knowledge-controls', 'knowledge-grid', 'predict-top', 'predict-grid'];
  function buildAll() {
    charts.forEach((c) => { cancelAnimationFrame(c.raf); clearTimeout(c.timer); });
    charts = [];
    hooks = [];
    CONTAINERS.forEach((id) => document.getElementById(id).replaceChildren());
    buildOverview();
    buildNB();
    buildDT();
    buildSVM();
    buildCompare();
    buildThreshold();
    buildKnowledge();
    buildPredict();
    onState(['keyword', 'depth'], renderInspector);
    renderInspector();
  }

  const btn = document.getElementById('theme-btn');
  const order = ['auto', 'light', 'dark'];
  let theme = 'auto';
  try { theme = localStorage.getItem('wine-theme') || 'auto'; } catch (e) { /* storage unavailable */ }
  const applyTheme = () => {
    if (theme === 'auto') document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', theme);
    btn.textContent = `Theme: ${theme}`;
  };
  applyTheme();
  btn.addEventListener('click', () => {
    theme = order[(order.indexOf(theme) + 1) % order.length];
    try { localStorage.setItem('wine-theme', theme); } catch (e) { /* storage unavailable */ }
    document.documentElement.classList.add('theme-anim');
    setTimeout(() => document.documentElement.classList.remove('theme-anim'), 500);
    applyTheme();
    buildAll();
  });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { if (theme === 'auto') buildAll(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && state.keyword) selectKeyword(state.keyword); });

  buildAll();
  setupMotion();
  setupDecor();

  let rt;
  window.addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(resizeAll, 150); });

  const links = [...document.querySelectorAll('nav.tabs a')];
  const sections = [...document.querySelectorAll('main section')];
  // sliding underline under the active section link
  const nav = document.querySelector('nav.tabs');
  const ind = h('span', { class: 'tab-ind', 'aria-hidden': 'true' });
  nav.querySelector('.wrap').append(ind);
  nav.classList.add('has-ind');
  const moveInd = () => {
    const a = links.find((l) => l.classList.contains('active'));
    if (!a) return;
    ind.style.width = a.offsetWidth + 'px';
    ind.style.transform = `translateX(${a.offsetLeft}px)`;
    const sec = document.querySelector(a.getAttribute('href'));
    if (sec) ind.style.setProperty('--ind', getComputedStyle(sec).getPropertyValue('--accent').trim());
  };
  window.addEventListener('resize', moveInd);
  const markActive = () => {
    let current = sections[0];
    sections.forEach((sec) => { if (sec.getBoundingClientRect().top <= 120) current = sec; });
    if (innerHeight + scrollY >= document.documentElement.scrollHeight - 4) current = sections[sections.length - 1];
    links.forEach((l) => l.classList.toggle('active', l.getAttribute('href') === '#' + current.id));
    moveInd();
    const a = links.find((l) => l.classList.contains('active'));
    const bar = a && a.parentElement;
    if (bar && bar.scrollWidth > bar.clientWidth) {
      if (a.offsetLeft < bar.scrollLeft) bar.scrollLeft = a.offsetLeft - 8;
      else if (a.offsetLeft + a.offsetWidth > bar.scrollLeft + bar.clientWidth) bar.scrollLeft = a.offsetLeft + a.offsetWidth - bar.clientWidth + 8;
    }
  };
  window.addEventListener('scroll', markActive, { passive: true });
  markActive();
})();
