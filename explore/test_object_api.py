import csv
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
from http.server import ThreadingHTTPServer

from object_api import API, build, clean_proposals, csv_cell, year


class ObjectAPITest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.site = Path(cls.temp.name) / 'site'
        cls.state = Path(cls.temp.name) / 'index'
        layers = cls.site / 'layers/snapshot'
        layers.mkdir(parents=True)
        polygon = {'type': 'FeatureCollection', 'metadata': {'source': 'https://example.org'}, 'features': [
            {'type': 'Feature', 'properties': {'id': '1', 'name': 'Rúa'}, 'geometry': {
                'type': 'Polygon', 'coordinates': [[[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]]]}}]}
        (cls.site / 'barrios.geojson').write_text(json.dumps(polygon))
        (cls.site / 'data.json').write_text(json.dumps({'meta': {'city': 'Test'}, 'signs': [
            {'id': 'legacy-a', 'source_crop_id': 'a', 'src': 'a', 'la': 1, 'lo': 1, 's': .5, 'cluster_id': 'C'},
            {'id': 'legacy-only', 'src': 'a', 'la': 1, 'lo': 1, 's': .5}]}))
        def feature(identity, **props):
            lon = props.pop('lon', 1)
            return {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [lon, 1]}, 'properties': {
                'id': identity, 'source': 'apple', 'date': '2026-01-01', 'width': 50, 'height': 50,
                'crop': '/proposal-assets/' + identity + '.jpg', 'raw': '/proposal-context/' + identity,
                'score': .8, 'text': '', **props}}
        rows = [feature('a', duplicate_group='group-ab', group_rep='a', text='Rúa Café'),
                feature('b', score=.9, duplicate_group='group-ab', group_rep='a', source='gsv', date='2020-02', text='Farmacia', lon=3),
                feature('dup', duplicate_of='a'), feature('unknown', date='2020-02-30')]
        rows += [feature(f'single-{i:03}', text='Shop') for i in range(65)]
        (layers / 'store_sign.geojson').write_text(json.dumps({'type': 'FeatureCollection', 'features': rows}))
        cls.manifest = {'snapshot_id': 'snapshot', 'city': 'Test', 'categories': [
            {'id': 'store_sign', 'url': 'layers/snapshot/store_sign.geojson', 'label': 'Shop signs', 'color': '#abc', 'count': len(rows)}]}
        (cls.site / 'layers/manifest.json').write_text(json.dumps(cls.manifest))
        build(cls.site, cls.state)
        cls.api = API(cls.state, cls.site, origin='https://viewer.test')
        cls.server = ThreadingHTTPServer(('127.0.0.1', 0), cls.api.handler())
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.origin = f'http://127.0.0.1:{cls.server.server_port}'

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temp.cleanup()

    def request(self, route='query', **params):
        params = {'layers': 'store_sign', 'confidence': .5, **params}
        with urlopen(self.origin + '/api/' + route + '?' + urlencode(params)) as response:
            body = response.read().decode('utf-8-sig')
        return json.loads(body) if route != 'export' or params.get('format') in ('geojson', 'manifest') else body

    def test_confidence_applies_to_counts_details_and_exports(self):
        self.assertEqual(self.request(confidence=.65)['total'], 67)
        self.assertEqual(self.request(confidence=.95)['total'], 0)
        self.assertEqual(self.request('export', format='manifest', confidence=.95)['record_count'], 0)
        self.assertEqual(self.request('object', id='group-ab', confidence=.85)['id'], 'b')
        self.assertEqual(self.request(confidence=.85)['total'], 1)
        self.assertEqual(self.request('export', format='geojson', confidence=.85)['features'][0]['properties']['member_ids'], ['b'])
        for value in ['nan', '.49', '1.1']:
            with self.assertRaises(HTTPError) as error:
                self.request(confidence=value)
            self.assertEqual(error.exception.code, 400)
        with self.assertRaises(HTTPError) as error:
            self.request('object', id='legacy-only', confidence=.65)
        self.assertEqual(error.exception.code, 404)

    def test_light_size_and_overlap_filter(self):
        def f(identity, box, score=.8, **extra):
            return {'properties': dict(id=identity, source='apple', view_id='one', category='door',
                                       bx=box, score=score, width=100, height=100, **extra)}
        rows = [f('best', [0, 0, .5, .5], .9), f('larger', [0, 0, .55, .55]),
                f('adjacent', [.5, 0, 1, .5]),
                f('shifted', [.15, 0, .65, .5]),
                f('contained', [.05, .05, .35, .35]),
                f('nested-sign', [0, 0, .2, .2], .7),
                f('tiny', [.8, .8, 1, 1], object_width=12, object_height=100)]
        other_category = f('graffiti', [0, 0, .5, .5]); other_category['properties']['category'] = 'graffiti'
        other_view = f('other-view', [0, 0, .5, .5]); other_view['properties']['view_id'] = 'two'
        rows.extend([other_category, other_view])
        self.assertEqual({r['properties']['id'] for r in clean_proposals(rows)},
                         {'best', 'adjacent', 'nested-sign', 'graffiti', 'other-view'})

    def test_page_counts_and_no_duplicate_observations(self):
        first = self.request(bbox='0,0,4,2', cols=1, rows=1)
        self.assertEqual(first['total'], 68)
        self.assertEqual(first['visible'], 68)
        self.assertEqual(sum(c['count'] for c in first['cells']), 68)
        self.assertEqual(len(first['items']), 48)
        self.assertTrue(all('source' in item for item in first['items']))
        self.assertEqual(next(item for item in first['items'] if item['id'] == 'a')['d'], '2026-01-01')
        self.assertTrue(first['has_more'])
        second = self.request(offset=48, view='wall')
        self.assertEqual(len(second['items']), 20)
        self.assertFalse(second['has_more'])
        self.assertFalse({v['id'] for v in first['items']} & {v['id'] for v in second['items']})
        self.assertEqual(self.request(group=0)['total'], 69)
        self.assertEqual(self.request(layers='')['total'], 0)

    def test_group_filters_choose_retained_viewpoint(self):
        obj = self.request('object', id='group-ab')
        self.assertEqual(obj['id'], 'a')
        self.assertEqual([v['id'] for v in obj['views']], ['a', 'b'])
        self.assertTrue(obj['legacy_record'])
        self.assertEqual(obj['crop'], '/proposal-assets/a.jpg')
        self.assertEqual(obj['barrio_id'], '1')
        self.assertEqual(self.request('object', id='a', source='gsv')['id'], 'b')
        self.assertEqual(self.request('object', id='group-ab', **{'from': 2020, 'to': 2020})['id'], 'b')
        self.assertEqual(self.request('object', id='group-ab', barrio='outside')['id'], 'b')
        self.assertEqual(self.request('object', id='b', group=0)['view_count'], 1)
        # The nonrepresentative view cannot drag a group into a map selection.
        self.assertEqual(self.request(bbox='2.5,0,3.5,2', area=1)['total'], 0)
        self.assertEqual(self.request(bbox='2.5,0,3.5,2', area=1, source='gsv')['total'], 1)

    def test_search_keeps_context_and_normalizes_accents(self):
        for query in ['farmacia', 'CAFE', 'afé', 'fé']:
            result = self.request(q=query)
            self.assertEqual(result['total'], 1)
            self.assertEqual(result['items'][0]['view_count'], 2)
        self.assertEqual(self.request(q='farmacia', source='apple')['total'], 0)
        self.assertEqual(self.request(q='farmacia', source='gsv')['total'], 1)
        self.assertEqual(self.request(unknown=0)['total'], 66)

    def test_exports_stream_all_matching_rows_and_views(self):
        rows = [json.loads(line) for line in self.request('export', format='jsonl').splitlines()]
        self.assertEqual(len(rows), 68)
        grouped = next(r for r in rows if r['id'] == 'group-ab')
        self.assertEqual(grouped['member_ids'], ['a', 'b'])
        self.assertEqual(grouped['crop_url'], 'https://viewer.test/proposal-assets/a.jpg')
        geo = self.request('export', format='geojson', q='farmacia')
        self.assertEqual(len(geo['features']), 1)
        self.assertEqual(geo['metadata']['record_count'], 1)
        self.assertEqual(self.request('export', format='manifest')['record_count'], 68)
        self.assertEqual(len(list(csv.reader(io.StringIO(self.request('export', format='csv'))))), 69)
        self.assertEqual(self.request('export', format='geojson', bbox='2.5,0,3.5,2', area=1)['features'], [])

    def test_validation_and_expired_snapshot(self):
        for params, status in [({'bbox': 'nan,0,2,2'}, 400), ({'layers': 'invalid'}, 400),
                               ({'from': 'bad'}, 400), ({'snapshot': '0'*24}, 409)]:
            with self.assertRaises(HTTPError) as error:
                self.request(**params)
            self.assertEqual(error.exception.code, status)
        with self.assertRaises(HTTPError) as error:
            self.request('object', id='missing')
        self.assertEqual(error.exception.code, 404)
        self.assertIsNone(year('2023-02-29'))
        self.assertEqual(year('2024-02-29'), 2024)
        self.assertTrue(csv_cell(' =SUM(A1)').startswith('"\''))

    def test_failed_build_preserves_current_snapshot(self):
        current = (self.state / 'current.json').read_bytes()
        path = self.site / 'layers/manifest.json'
        before = path.read_text()
        try:
            path.write_text(json.dumps({**self.manifest, 'snapshot_id': 'broken', 'categories': [
                {**self.manifest['categories'][0], 'url': '../../secret'}]}))
            with self.assertRaises(ValueError):
                build(self.site, self.state)
            self.assertEqual((self.state / 'current.json').read_bytes(), current)
            self.assertEqual(self.request()['total'], 68)
        finally:
            path.write_text(before)


if __name__ == '__main__':
    unittest.main()
