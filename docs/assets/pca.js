(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const chart = $('pca-chart');
  const cache = new Map();
  let manifestPromise, libraryPromise, started = false, generation = 0, timer;
  let drawing = Promise.resolve();

  function json(path) {
    if (!cache.has(path)) {
      const promise = fetch(path).then(response => {
        if (!response.ok) throw new Error('Could not load the PCA data.');
        return response.json();
      }).catch(error => { cache.delete(path); throw error; });
      cache.set(path, promise);
    }
    return cache.get(path);
  }

  function library() {
    if (!libraryPromise) libraryPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'assets/vendor/plotly-gl3d-2.35.2.min.js';
      script.onload = resolve;
      script.onerror = () => {
        script.remove();
        libraryPromise = null;
        reject(new Error('Could not load the 3D viewer.'));
      };
      document.head.append(script);
    });
    return libraryPromise;
  }

  function escape(text) {
    return text.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
  }

  async function update() {
    const current = ++generation;
    const viewId = $('pca-view').value;
    const layer = Number($('pca-layer').value);
    $('pca-layer-value').textContent = layer;
    $('pca-pos').disabled = viewId !== 'bert-pos';
    $('pca-pos').parentElement.hidden = viewId !== 'bert-pos';
    if (!started) return;
    $('pca-status').textContent = 'Loading projection…';
    $('pca-error').hidden = true;
    chart.setAttribute('aria-busy', 'true');
    try {
      if (!manifestPromise) manifestPromise = json('assets/pca/manifest.json').catch(error => { manifestPromise = null; throw error; });
      const [manifest] = await Promise.all([manifestPromise, library()]);
      const view = manifest.views.find(v => v.id === viewId);
      if ($('pca-pos').options.length === 1) {
        for (const group of manifest.views.find(v => v.id === 'bert-pos').groups) {
          $('pca-pos').add(new Option(group.name, group.name));
        }
      }
      const data = await json(view.layers[layer].path);
      if (current !== generation) return;
      const selected = viewId === 'bert-pos' ? $('pca-pos').value : 'all';
      const traces = data.traces.filter(t => selected === 'all' || t.name === selected).map(t => ({
        type: 'scatter3d', mode: 'markers', name: t.name,
        x: t.x, y: t.y, z: t.z, text: t.text.map(escape),
        marker: {size: 2.5, opacity: .72, color: t.color},
        hovertemplate: '<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra>%{fullData.name}</extra>',
        showlegend: false,
      }));
      await drawing.catch(() => {});
      if (current !== generation) return;
      const camera = chart.dataset.view === viewId && chart._fullLayout ? structuredClone(chart._fullLayout.scene.camera) : {eye: {x: 1.6, y: 1.6, z: 1.3}};
      chart.hidden = false;
      const axis = name => ({title: {text: name}, showbackground: true, backgroundcolor: '#f7f9fb', gridcolor: '#d5dfe7', zerolinecolor: '#a6b8c7'});
      drawing = Plotly.react(chart, traces, {
        margin: {l: 35, r: 25, b: 35, t: 10},
        font: {family: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif', size: 14, color: '#202b35'},
        paper_bgcolor: '#ffffff',
        scene: {xaxis: axis('PC1'), yaxis: axis('PC2'), zaxis: axis('PC3'), aspectmode: 'auto', dragmode: 'orbit', camera},
        uirevision: viewId,
      }, {responsive: true, displaylogo: false, displayModeBar: false, scrollZoom: false});
      await drawing;
      if (current !== generation) return;
      const points = traces.reduce((n, t) => n + t.x.length, 0);
      chart.dataset.view = viewId;
      chart.dataset.layer = layer;
      chart.setAttribute('aria-label', `${view.label}, layer ${layer}, ${points.toLocaleString('en')} sampled word occurrences. Drag to rotate; hover to read a word.`);
      $('pca-status').textContent = `${view.label} · layer ${layer} · ${points.toLocaleString('en')} sampled word occurrences`;
      $('load-pca').hidden = true;
      const legend = $('pca-legend');
      legend.replaceChildren();
      legend.hidden = viewId !== 'bert-pos';
      if (!legend.hidden) {
        for (const trace of traces) {
          const item = document.createElement('span');
          const swatch = document.createElement('i');
          swatch.style.backgroundColor = trace.marker.color;
          item.append(swatch, document.createTextNode(trace.name));
          legend.append(item);
        }
      }
    } catch (error) {
      if (current !== generation) return;
      chart.hidden = true;
      $('pca-legend').hidden = true;
      $('pca-status').textContent = 'The projection could not be displayed.';
      $('pca-error').textContent = `${error.message} Please try again.`;
      $('pca-error').hidden = false;
      $('load-pca').hidden = false;
      $('load-pca').textContent = 'Retry 3D plot';
    } finally {
      if (current === generation) chart.setAttribute('aria-busy', 'false');
    }
  }

  $('load-pca').addEventListener('click', () => { started = true; update(); });
  $('pca-view').addEventListener('change', update);
  $('pca-pos').addEventListener('change', update);
  $('pca-layer').addEventListener('input', () => {
    $('pca-layer-value').textContent = $('pca-layer').value;
    ++generation;
    clearTimeout(timer);
    timer = setTimeout(update, 120);
  });
})();
