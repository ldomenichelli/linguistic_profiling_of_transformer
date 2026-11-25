#!/usr/bin/env python3
"""Check the website in Chrome through a local DevTools endpoint.

Start `make serve` and a headless Chrome instance with remote debugging, then run
this script. Requires websocket-client; does not run representation extraction.
"""
import argparse
import base64
import csv
import json
import time
import urllib.request
from pathlib import Path

import websocket


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8766/')
    parser.add_argument('--devtools', default='http://127.0.0.1:9225')
    parser.add_argument('--screenshots', type=Path)
    args = parser.parse_args()
    tabs = json.load(urllib.request.urlopen(args.devtools + '/json'))
    tab = next(t for t in tabs if t['type'] == 'page')
    ws = websocket.create_connection(tab['webSocketDebuggerUrl'], timeout=30)
    index = 0
    errors, failures = [], []

    def call(method, params=None):
        nonlocal index
        index += 1
        ws.send(json.dumps({'id': index, 'method': method, 'params': params or {}}))
        while True:
            message = json.loads(ws.recv())
            if message.get('method') == 'Runtime.exceptionThrown':
                errors.append(message['params'].get('exceptionDetails', {}))
            if message.get('method') == 'Network.responseReceived' and message['params']['response']['status'] >= 400:
                failures.append(message['params']['response']['url'])
            if message.get('id') == index:
                assert 'error' not in message, message
                return message.get('result', {})

    def js(expression):
        result = call('Runtime.evaluate', {'expression': expression, 'returnByValue': True, 'awaitPromise': True})
        assert 'exceptionDetails' not in result, result
        return result.get('result', {}).get('value')

    def wait(expression, timeout=25):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if js(expression):
                return
            time.sleep(.08)
        raise AssertionError('Timed out: ' + expression)

    def change(identifier, value, event='change'):
        js(f"document.getElementById({json.dumps(identifier)}).value={json.dumps(value)};document.getElementById({json.dumps(identifier)}).dispatchEvent(new Event({json.dumps(event)},{{bubbles:true}}))")

    def snapshot(name):
        if not args.screenshots:
            return
        args.screenshots.mkdir(parents=True, exist_ok=True)
        result = call('Page.captureScreenshot', {'format': 'png'})
        (args.screenshots / (name + '.png')).write_bytes(base64.b64decode(result['data']))

    call('Runtime.enable')
    call('Page.enable')
    call('Network.enable')
    call('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 1050, 'deviceScaleFactor': 1, 'mobile': False})
    call('Page.navigate', {'url': args.url})
    wait("document.querySelectorAll('.layer-chart').length===2")
    assert js("typeof CORPUS_EN==='undefined' && typeof CORPUS_IT==='undefined'")
    assert js("document.querySelectorAll('.interval-band').length") == 2
    assert js("document.querySelector('.chart-panel[data-model=gpt2]').textContent.includes('intervals are unavailable')")
    assert js("document.querySelectorAll('#selected-values tr').length") == 4
    assert js("[...document.querySelectorAll('a[href]')].every(a=>!a.href.includes('review-2.pdf'))")
    snapshot('desktop')
    source = Path(__file__).resolve().parents[1] / 'plots_main/plots_features/feature_contrasts/tables/pos_open_closed_bert_isoscore_groups.csv'
    row = next(r for r in csv.DictReader(source.open()) if r['layer'] == '12' and r['group_key'] == 'positive')
    value = js("Number(document.querySelector('#selected-values tr[data-model=bert][data-group=positive] td:nth-child(3)').textContent.replaceAll(',',''))")
    assert abs(value - float(row['mean'])) <= max(1e-6, abs(float(row['mean'])) * 1e-5)
    change('comparison', 'length_long_short')
    change('metric', 'nlid')
    change('layer', '3', 'input')
    assert js("document.querySelector('#values-caption').textContent") == 'Values at layer 3'
    assert js("new URL(location.href).searchParams.get('comparison')") == 'length_long_short'
    assert js("document.querySelectorAll('.interval-band').length") == 4
    js("document.querySelector('#intervals').click()")
    assert js("document.querySelectorAll('.interval-band').length") == 0
    js("document.querySelector('#intervals').click()")
    change('comparison', 'arity_high_low')
    change('metric', 'lid')
    arity_source = source.parent / 'arity_high_low_bert_LID_groups.csv'
    missing = next(r for r in csv.DictReader(arity_source.open()) if r['ci_method'] == 'no_source_uncertainty')
    change('layer', missing['layer'], 'input')
    assert js(f"document.querySelector('#selected-values tr[data-model=bert][data-group={missing['group_key']}] td:nth-child(4)').textContent") == 'Unavailable'
    change('comparison', 'verbal_valency_mid_low')
    assert js("document.querySelectorAll('.missing-series').length") == 2
    change('metric', 'iso')
    assert js("document.querySelectorAll('.missing-series').length") == 1
    assert js("document.querySelectorAll('.layer-chart').length") == 1
    js("document.querySelector('.missing-series button').click()")
    assert js("document.querySelector('#figure-dialog').open")
    assert js("document.querySelector('#figure-caption').textContent.includes('Verbal valency')")
    js("document.querySelector('#close-figure').click()")
    change('figure-feature', 'pos')
    change('figure-model', 'bert')
    change('figure-metric', 'lid')
    assert js("document.querySelectorAll('.figure').length") == 1
    js("document.querySelector('.figure-open').click()")
    assert js("document.querySelector('#figure-dialog').open")
    js("document.querySelector('#close-figure').click()")
    js("document.querySelector('#load-corpus').click()")
    wait("typeof CORPUS_EN!=='undefined' && !document.querySelector('#corpus-view').hidden")
    assert js("CORPUS_EN.records.length") == 10067
    assert js("typeof CORPUS_IT==='undefined'")
    assert js("document.querySelectorAll('.sentence-button').length") == 6
    change('text-search', 'the', 'input')
    time.sleep(.25)
    wait("document.querySelectorAll('.sentence-button').length>0")
    change('text-pos', 'DET')
    assert js("document.querySelector('.token-information').textContent.includes('DET')")
    js("document.querySelectorAll('.token')[1].click()")
    assert js("document.querySelectorAll('.token[aria-pressed=true]').length") == 1
    assert js("document.querySelector('.word-title').textContent") == js("document.querySelector('.token[aria-pressed=true]').textContent")
    change('text-search', 'no-such-token-xyz-987654321', 'input')
    time.sleep(.25)
    wait("document.querySelector('#sentence-page').textContent==='0 / 0'")
    assert js("document.querySelector('#sentence-detail').textContent.includes('No sentences')")
    assert js("document.querySelector('#download-sentences').disabled")
    change('corpus-language', 'it')
    wait("typeof CORPUS_IT!=='undefined' && !document.querySelector('#text-search').disabled")
    assert js("CORPUS_IT.records.length") == 12112
    assert js("document.querySelector('#corpus-status').textContent.includes('253,771')")
    assert js("document.querySelector('#corpus-source').href.includes('it_isdt')")
    snapshot('italian-texts')
    assert js("Promise.all([...document.querySelectorAll('a[href]')].filter(a=>a.origin===location.origin&&!a.hash).map(a=>fetch(a.href,{method:'HEAD'}).then(r=>r.ok))).then(a=>a.every(Boolean))")
    for width, height, name in [(390, 844, 'mobile'), (768, 1024, 'tablet')]:
        call('Emulation.setDeviceMetricsOverride', {'width': width, 'height': height, 'deviceScaleFactor': 1, 'mobile': True})
        js("window.scrollTo({top:0,behavior:'instant'})")
        time.sleep(.15)
        assert js("document.documentElement.scrollWidth<=innerWidth"), name
        snapshot(name)
        js("document.querySelector('#texts').scrollIntoView({behavior:'instant'})")
        time.sleep(.15)
        assert js("document.documentElement.scrollWidth<=innerWidth"), name + ' texts'
        snapshot(name + '-texts')
    assert not errors, errors
    assert not failures, failures
    ws.close()
    print(json.dumps({'status': 'passed', 'checks': ['source values', 'missing uncertainty', 'layer and metric controls', 'digitized-result omissions', 'figure filters and viewer', 'lazy corpus loading', 'English search and POS', 'Italian annotations', 'empty results', 'asset links', 'mobile and tablet layout'], 'javascript_errors': len(errors), 'failed_requests': len(failures)}, indent=2))


if __name__ == '__main__':
    main()
