(() => {
  'use strict';
  const D = window.SITE_DATA;
  const $ = id => document.getElementById(id);
  const colours = {negative: '#24659c', positive: '#c65b2d'};
  const labels = {iso: 'IsoScore', lid: 'Linear ID', nlid: 'Nonlinear ID'};
  const models = {bert: 'BERT', gpt2: 'GPT-2'};
  const params = new URLSearchParams(location.search);
  const state = {
    comparison: D.comparisons.some(c => c.id === params.get('comparison')) ? params.get('comparison') : 'pos_open_closed',
    metric: ['iso', 'lid', 'nlid'].includes(params.get('metric')) ? params.get('metric') : 'iso',
    layer: Math.round(Math.max(0, Math.min(12, Number(params.get('layer') ?? 12) || 0))),
    figureLimit: 6, corpus: null, matches: [], page: 0, sentence: null, token: 0,
  };
  const loading = new Map();
  const searchIndexes = new Map();

  function el(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = text;
    if (className) node.className = className;
    return node;
  }
  function link(text, href, download = false) {
    const node = el('a', text);
    node.href = href;
    if (download) node.setAttribute('download', '');
    return node;
  }
  function button(text, fn, className) {
    const node = el('button', text, className);
    node.type = 'button';
    node.addEventListener('click', fn);
    return node;
  }
  function format(value) {
    if (value === null || value === undefined) return 'Unavailable';
    return Number(value).toLocaleString('en', {maximumSignificantDigits: 6});
  }
  function hasInterval(point) {
    return point.intervalMethod !== 'no_source_uncertainty' && point.low !== null && point.high !== null;
  }
  function save(name, value, type) {
    const url = URL.createObjectURL(new Blob([value], {type}));
    const node = link(name, url, true);
    node.download = name;
    node.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  function updateAddress() {
    const url = new URL(location.href);
    for (const key of ['comparison', 'metric', 'layer']) url.searchParams.set(key, state[key]);
    history.replaceState(null, '', url);
  }
  function figureFor(model) {
    const feature = state.comparison.startsWith('verbal_valency') ? 'verbal_valency' : state.comparison.split('_')[0];
    const mapped = {head: 'head', relation: 'relation'}[feature] || feature;
    return D.figures.find(f => f.feature === mapped && f.model === model && f.metric === state.metric);
  }

  function chart(series) {
    const width = 520, height = 290;
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', `${models[series.model]}: ${labels[series.metric]} from layer 0 to 12. Values are also available in the table below.`);
    svg.classList.add('layer-chart');
    function shape(tag, attrs, text) {
      const node = document.createElementNS(svg.namespaceURI, tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      if (text !== undefined) node.textContent = text;
      svg.append(node);
      return node;
    }
    const showIntervals = $('intervals').checked;
    const values = series.points.flatMap(p => showIntervals && hasInterval(p) ? [p.mean, p.low, p.high] : [p.mean]);
    let low = Math.min(...values), high = Math.max(...values);
    const padding = (high - low || Math.abs(high) || 1) * 0.1;
    low -= padding; high += padding;
    const x = layer => 60 + layer / 12 * 435;
    const y = value => 240 - (value - low) / (high - low) * 210;
    for (let n = 0; n <= 4; n++) {
      const value = low + (high - low) * n / 4;
      shape('line', {x1: 60, x2: 495, y1: y(value), y2: y(value), class: 'grid-line'});
      shape('text', {x: 51, y: y(value) + 4, 'text-anchor': 'end', class: 'axis-text'}, Number(value.toPrecision(3)).toLocaleString('en'));
    }
    for (let layer = 0; layer <= 12; layer += 2) {
      shape('text', {x: x(layer), y: 260, 'text-anchor': 'middle', class: 'axis-text'}, layer);
    }
    shape('text', {x: 275, y: 283, 'text-anchor': 'middle', class: 'axis-label'}, 'Layer');
    shape('text', {x: 60, y: 15, class: 'axis-label'}, labels[series.metric]);
    shape('line', {x1: x(state.layer), x2: x(state.layer), y1: 28, y2: 240, class: 'selected-layer'});
    for (const group of ['negative', 'positive']) {
      const points = series.points.filter(p => p.group === group).sort((a, b) => a.layer - b.layer);
      if (showIntervals && points.every(hasInterval)) {
        const outline = [...points.map(p => [x(p.layer), y(p.high)]), ...[...points].reverse().map(p => [x(p.layer), y(p.low)])];
        shape('polygon', {points: outline.map(p => p.join(',')).join(' '), fill: colours[group], class: 'interval-band'});
      } else if (showIntervals) {
        for (const p of points.filter(hasInterval)) {
          shape('line', {x1: x(p.layer), x2: x(p.layer), y1: y(p.low), y2: y(p.high), stroke: colours[group], 'stroke-width': 1.2});
          for (const value of [p.low, p.high]) shape('line', {x1: x(p.layer) - 3, x2: x(p.layer) + 3, y1: y(value), y2: y(value), stroke: colours[group], 'stroke-width': 1.2});
        }
      }
      shape('polyline', {points: points.map(p => `${x(p.layer)},${y(p.mean)}`).join(' '), stroke: colours[group], class: 'curve'});
      for (const point of points) {
        const circle = shape('circle', {cx: x(point.layer), cy: y(point.mean), r: point.layer === state.layer ? 5 : 3, fill: colours[group], class: 'point', 'data-layer': point.layer});
        const title = document.createElementNS(svg.namespaceURI, 'title');
        title.textContent = `${point.label} · layer ${point.layer}: ${point.mean}`;
        circle.append(title);
        circle.addEventListener('click', () => {
          state.layer = point.layer;
          $('layer').value = point.layer;
          renderResults();
          updateAddress();
        });
      }
    }
    return svg;
  }

  function renderResults() {
    $('layer-value').textContent = state.layer;
    $('values-caption').textContent = `Values at layer ${state.layer}${state.layer === 0 ? ' · embeddings' : ''}`;
    $('charts').replaceChildren();
    $('selected-values').replaceChildren();
    for (const model of ['bert', 'gpt2']) {
      const series = D.series.find(s => s.comparison === state.comparison && s.model === model && s.metric === state.metric);
      const panel = el('article', null, 'chart-panel');
      panel.dataset.model = model;
      const heading = el('div', null, 'chart-heading');
      heading.append(el('h3', models[model]));
      panel.append(heading);
      if (!series) {
        const missing = el('div', null, 'missing-series');
        missing.append(el('p', 'A numeric source table is not available for this comparison.'));
        const figure = figureFor(model);
        if (figure) missing.append(button('View saved figure', () => openFigure(figure)));
        panel.append(missing);
        $('charts').append(panel);
        continue;
      }
      heading.append(link('Source CSV', series.csv, true));
      const legend = el('div', null, 'legend');
      for (const group of ['negative', 'positive']) {
        const point = series.points.find(p => p.group === group);
        const item = el('span');
        const swatch = el('i', null, 'legend-mark');
        swatch.style.background = colours[group];
        item.append(swatch, document.createTextNode(point.label));
        legend.append(item);
      }
      panel.append(legend, chart(series));
      if (series.excludesFirst) panel.append(el('p', 'Sentence-initial words excluded.', 'panel-note'));
      if (series.points.every(p => !hasInterval(p))) panel.append(el('p', 'Uncertainty intervals are unavailable in this source.', 'panel-note'));
      else if (series.points.some(p => !hasInterval(p))) panel.append(el('p', 'Some source values do not include uncertainty intervals.', 'panel-note'));
      const details = el('details', null, 'all-layers');
      details.append(el('summary', 'View all layer values'));
      const scroll = el('div', null, 'table-scroll');
      const table = el('table');
      table.append(el('caption', `${models[model]} · ${labels[state.metric]}`));
      const head = el('thead'), tr = el('tr');
      for (const name of ['Layer', 'Group', 'Mean', 'Lower', 'Upper']) { const th = el('th', name); th.scope = 'col'; tr.append(th); }
      head.append(tr); table.append(head);
      const body = el('tbody');
      for (const p of [...series.points].sort((a, b) => a.layer - b.layer || a.group.localeCompare(b.group))) {
        const row = el('tr');
        for (const value of [p.layer, p.label, format(p.mean), format(hasInterval(p) ? p.low : null), format(hasInterval(p) ? p.high : null)]) row.append(el('td', value));
        body.append(row);
      }
      table.append(body); scroll.append(table); details.append(scroll); panel.append(details);
      $('charts').append(panel);
      for (const p of series.points.filter(p => p.layer === state.layer)) {
        const row = el('tr');
        row.dataset.model = model; row.dataset.group = p.group;
        for (const value of [models[model], p.label, format(p.mean), format(hasInterval(p) ? p.low : null), format(hasInterval(p) ? p.high : null), p.n.toLocaleString('en')]) row.append(el('td', value));
        $('selected-values').append(row);
      }
    }
  }

  function openFigure(figure) {
    $('large-figure').src = figure.png;
    $('large-figure').alt = figure.title;
    $('figure-caption').textContent = figure.title;
    $('figure-download').href = figure.pdf;
    $('figure-dialog').showModal();
  }
  function renderFigures() {
    const feature = $('figure-feature').value, model = $('figure-model').value, metric = $('figure-metric').value;
    const figures = D.figures.filter(f => (feature === 'all' || f.feature === feature) && (model === 'all' || f.model === model || f.model === 'both') && (metric === 'all' || f.metric === metric));
    $('figure-grid').replaceChildren();
    for (const figure of figures.slice(0, state.figureLimit)) {
      const item = el('figure', null, 'figure');
      const open = button('', () => openFigure(figure), 'figure-open');
      open.setAttribute('aria-label', 'Enlarge ' + figure.title);
      const image = el('img');
      image.src = figure.png; image.alt = figure.title; image.loading = 'lazy';
      open.append(image);
      const caption = el('figcaption');
      caption.append(el('span', figure.title), link('Figure PDF', figure.pdf, true));
      item.append(open, caption); $('figure-grid').append(item);
    }
    $('figure-count').textContent = `${Math.min(state.figureLimit, figures.length)} of ${figures.length} figures`;
    $('more-figures').hidden = state.figureLimit >= figures.length;
  }

  function loadScript(path) {
    if (!loading.has(path)) loading.set(path, new Promise((resolve, reject) => {
      const node = document.createElement('script');
      node.src = path;
      node.onload = resolve;
      node.onerror = () => { node.remove(); loading.delete(path); reject(new Error('The corpus could not be loaded. Please try again.')); };
      document.head.append(node);
    }));
    return loading.get(path);
  }
  async function loadCorpus() {
    const id = $('corpus-language').value;
    const metadata = D.corpora.find(c => c.id === id);
    $('corpus-error').hidden = true;
    $('corpus-status').textContent = `Loading ${metadata.label}…`;
    $('load-corpus').disabled = true;
    $('text-search').disabled = $('text-pos').disabled = true;
    $('corpus-view').hidden = true;
    try {
      await loadScript(metadata.path);
      if ($('corpus-language').value !== id) return;
      state.corpus = window['CORPUS_' + id.toUpperCase()];
      state.page = 0; state.sentence = null;
      if (!searchIndexes.has(id)) searchIndexes.set(id, state.corpus.records.map(r => (r[0] + ' ' + r[1].join(' ')).toLocaleLowerCase(id)));
      $('text-pos').replaceChildren();
      for (const value of ['all', ...metadata.pos]) { const option = el('option', value === 'all' ? 'All word classes' : value); option.value = value; $('text-pos').append(option); }
      $('text-search').value = '';
      $('text-search').disabled = $('text-pos').disabled = false;
      $('load-corpus').hidden = true;
      $('corpus-view').hidden = false;
      $('corpus-source').href = D.repository + '/blob/main/' + metadata.source;
      filterSentences();
    } catch (error) {
      $('corpus-error').textContent = error.message;
      $('corpus-error').hidden = false;
      $('corpus-status').textContent = 'Corpus unavailable.';
      $('load-corpus').hidden = false;
    } finally {
      $('load-corpus').disabled = false;
    }
  }
  function matchingToken(record, index) {
    const q = $('text-search').value.trim().toLocaleLowerCase(state.corpus.id);
    const pos = $('text-pos').value;
    const sentenceQuery = q.includes(' ') || record[0].toLocaleLowerCase(state.corpus.id).includes(q);
    return (pos === 'all' || record[4][index] === pos) && (!q || sentenceQuery || record[1][index].toLocaleLowerCase(state.corpus.id).includes(q));
  }
  function filterSentences() {
    if (!state.corpus) return;
    const q = $('text-search').value.trim().toLocaleLowerCase(state.corpus.id);
    const pos = $('text-pos').value;
    const search = searchIndexes.get(state.corpus.id);
    state.matches = state.corpus.records.map((r, i) => i).filter(i => (!q || search[i].includes(q)) && (pos === 'all' || state.corpus.records[i][1].some((_, token) => matchingToken(state.corpus.records[i], token))));
    state.page = 0;
    state.sentence = state.matches[0] ?? null;
    renderSentences();
  }
  function renderSentences() {
    const metadata = D.corpora.find(c => c.id === state.corpus.id);
    $('corpus-status').textContent = `${state.matches.length.toLocaleString('en')} / ${metadata.sentences.toLocaleString('en')} sentences · ${metadata.tokens.toLocaleString('en')} tokens in the stored corpus`;
    $('sentence-list').replaceChildren();
    for (const index of state.matches.slice(state.page * 6, state.page * 6 + 6)) {
      const record = state.corpus.records[index];
      const text = record[1].join(' ');
      const item = button(text.length > 210 ? text.slice(0, 210) + '…' : text, () => {
        state.sentence = index;
        renderSentences();
        if (matchMedia('(max-width:800px)').matches) $('sentence-detail').scrollIntoView({block: 'start', behavior: 'auto'});
      }, 'sentence-button' + (state.sentence === index ? ' active' : ''));
      item.setAttribute('aria-pressed', String(state.sentence === index));
      item.append(el('small', record[0]));
      $('sentence-list').append(item);
    }
    const pages = Math.ceil(state.matches.length / 6);
    $('sentence-page').textContent = pages ? `${state.page + 1} / ${pages}` : '0 / 0';
    $('previous-sentences').disabled = state.page === 0;
    $('next-sentences').disabled = state.page + 1 >= pages;
    $('download-sentences').disabled = state.matches.length === 0;
    renderSentence();
  }
  function renderSentence() {
    $('sentence-detail').replaceChildren();
    if (state.sentence === null) {
      $('sentence-detail').append(el('p', 'No sentences match these filters.'));
      return;
    }
    const record = state.corpus.records[state.sentence];
    state.token = Math.max(0, record[1].findIndex((_, i) => matchingToken(record, i)));
    $('sentence-detail').append(el('h3', 'Selected sentence'), el('p', record[0], 'sentence-source'));
    const ribbon = el('div', null, 'token-ribbon');
    ribbon.setAttribute('aria-label', 'Words in the selected sentence');
    record[1].forEach((word, index) => {
      const filtering = $('text-search').value.trim() || $('text-pos').value !== 'all';
      const token = button(word, () => { state.token = index; renderToken(); }, 'token' + (filtering && matchingToken(record, index) ? ' match' : ''));
      token.dataset.index = index;
      token.setAttribute('aria-pressed', String(index === state.token));
      ribbon.append(token);
    });
    const information = el('div', null, 'token-information');
    information.setAttribute('aria-live', 'polite');
    $('sentence-detail').append(ribbon, information);
    renderToken();
  }
  function renderToken() {
    const record = state.corpus.records[state.sentence], index = state.token;
    const target = $('sentence-detail').querySelector('.token-information');
    target.replaceChildren(el('h4', record[1][index], 'word-title'));
    const list = el('dl', null, 'token-details');
    for (const [label, value] of [['Position', record[2][index]], ['Characters', record[3][index]], ['Part of speech', record[4][index]], ['Head distance', record[5][index]], ['Dependency relation', record[6][index]], ['Dependents', record[7][index]]]) {
      const pair = el('div'); pair.append(el('dt', label), el('dd', value)); list.append(pair);
    }
    target.append(list);
    $('sentence-detail').querySelectorAll('.token').forEach(node => node.setAttribute('aria-pressed', String(Number(node.dataset.index) === index)));
  }

  for (const comparison of D.comparisons) { const option = el('option', comparison.label); option.value = comparison.id; $('comparison').append(option); }
  $('comparison').value = state.comparison; $('metric').value = state.metric; $('layer').value = state.layer;
  for (const id of ['comparison', 'metric', 'intervals']) $(id).addEventListener('change', () => {
    state.comparison = $('comparison').value; state.metric = $('metric').value; renderResults(); updateAddress();
  });
  $('layer').addEventListener('input', () => { state.layer = Number($('layer').value); renderResults(); updateAddress(); });
  for (const id of ['figure-feature', 'figure-model', 'figure-metric']) $(id).addEventListener('change', () => { state.figureLimit = 6; renderFigures(); });
  $('more-figures').addEventListener('click', () => { state.figureLimit += 6; renderFigures(); });
  $('close-figure').addEventListener('click', () => $('figure-dialog').close());
  $('figure-dialog').addEventListener('click', event => { if (event.target === $('figure-dialog')) $('figure-dialog').close(); });
  $('load-corpus').addEventListener('click', loadCorpus);
  $('corpus-language').addEventListener('change', loadCorpus);
  let searchTimer;
  $('text-search').addEventListener('input', () => { clearTimeout(searchTimer); searchTimer = setTimeout(filterSentences, 180); });
  $('text-pos').addEventListener('change', filterSentences);
  for (const [id, step] of [['previous-sentences', -1], ['next-sentences', 1]]) $(id).addEventListener('click', () => {
    state.page += step; state.sentence = state.matches[state.page * 6] ?? null; renderSentences();
  });
  $('download-sentences').addEventListener('click', () => save(`sentences-${state.corpus.id}.json`, JSON.stringify({corpus: state.corpus.id, columns: state.corpus.columns, attribution: state.corpus.attribution, records: state.matches.map(i => state.corpus.records[i])}), 'application/json'));
  renderResults();
  renderFigures();
})();
