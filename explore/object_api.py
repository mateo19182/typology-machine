"""Indexed browsing of immutable published snapshots. Standard-library only."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import fcntl
from functools import lru_cache
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import threading
import unicodedata
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'data/coruna/site'
STATE = Path.home() / '.local/state/typology-machine/query'


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def normalize(value):
    return ''.join(c for c in unicodedata.normalize('NFD', str(value or '')) if not unicodedata.combining(c)).lower()


def mercator(lat):
    return math.degrees(math.asinh(math.tan(math.radians(max(-85, min(85, lat))))))


def latitude(y):
    return math.degrees(math.atan(math.sinh(math.radians(y))))


def ring_contains(ring, x, y):
    inside = False
    for a, b in zip(ring, ring[1:] + ring[:1]):
        ax, ay = a
        bx, by = b
        cross = (x-ax)*(by-ay)-(y-ay)*(bx-ax)
        if abs(cross) < 1e-12 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
        if (ay > y) != (by > y) and x < (bx-ax)*(y-ay)/(by-ay)+ax:
            inside = not inside
    return inside


class Barrios:
    def __init__(self, data):
        self.entries = []
        for f in sorted(data['features'], key=lambda f: f['properties']['name']):
            polygons = [f['geometry']['coordinates']] if f['geometry']['type'] == 'Polygon' else f['geometry']['coordinates']
            points = [p for poly in polygons for p in poly[0]]
            bounds = (min(p[0] for p in points), min(p[1] for p in points), max(p[0] for p in points), max(p[1] for p in points))
            self.entries.append((str(f['properties']['id']), f['properties']['name'], bounds, polygons))

    @lru_cache(maxsize=250000)
    def find(self, x, y):
        for identity, name, (w, s, e, n), polygons in self.entries:
            if w <= x <= e and s <= y <= n and any(ring_contains(poly[0], x, y) and not any(ring_contains(h, x, y) for h in poly[1:]) for poly in polygons):
                return identity, name
        return None, None


def legacy_rows(data):
    return {s.get('source_crop_id', s['id']): {
        **s, 'category': 'store_sign', 'legacy_record': True, 'observation_id': s.get('source_crop_id'),
        'id': s.get('source_crop_id', s['id']), 'score': s.get('score', s.get('s', 0)),
        'crop': f"img/{s['id']}.jpg", 'raw': f"ctx/{s['id']}.jpg" if s.get('ctx') else None,
        'source': 'gsv' if s['src'] == 'g' else 'apple', 'location_basis': 'estimated_object', 'status': 'legacy_detection',
    } for s in data['signs']}


def clean_proposals(features):
    """Light size gate and same-view, same-category NMS; originals remain intact."""
    views = {}
    for feature in features:
        p = feature['properties']
        if p.get('duplicate_of'):
            continue
        w, h = p.get('object_width', p['width']), p.get('object_height', p['height'])
        if min(w, h) < 24 or w * h < 1024:
            continue
        key = (p.get('source'), p.get('view_id'), p.get('category'))
        # Older exports without view geometry cannot be safely compared.
        if not p.get('view_id') or not p.get('bx'):
            yield feature
            continue
        views.setdefault(key, []).append(feature)
    def area(b):
        return max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    for candidates in views.values():
        kept = []
        for feature in sorted(candidates, key=lambda f: (-f['properties']['score'], -area(f['properties']['bx']), f['properties']['id'])):
            box = feature['properties']['bx']
            duplicate = False
            for other in kept:
                intersection = area([max(box[0], other[0]), max(box[1], other[1]), min(box[2], other[2]), min(box[3], other[3])])
                union = area(box) + area(other) - intersection
                smaller, larger = sorted((area(box), area(other)))
                # Catch both shifted boxes and near-contained alternative extents.
                # Keep small nested objects: they can be separate real details.
                if union > 0 and (intersection / union >= .45 or
                                  (smaller > 0 and intersection / smaller >= .85 and smaller / larger >= .3)):
                    duplicate = True
                    break
            if not duplicate:
                kept.append(box)
                yield feature


def published_rows(site, manifest, legacy):
    remaining = legacy_rows(legacy)
    for category in manifest['categories']:
        path = (site / category['url']).resolve()
        if not path.is_relative_to((site / 'layers').resolve()):
            raise ValueError('Invalid published layer path')
        print('Indexing', category['id'], flush=True)
        features = json.loads(path.read_text())['features']
        legacy_matches = {f['properties']['id']: remaining.pop(f['properties']['id'], {}) for f in features}
        for feature in clean_proposals(features):
            p = feature['properties']
            if p.get('duplicate_of'):
                continue
            lon, lat = feature['geometry']['coordinates']
            row = {**legacy_matches.get(p['id'], {}), **p, 'category': category['id'], 'la': lat, 'lo': lon,
                   'w': p['width'], 'h': p['height'], 'd': p['date'], 'u': p.get('provider_url'),
                   'src': 'g' if p['source'] == 'gsv' else 'a', 'proposal': True,
                   'snapshot_id': manifest['snapshot_id']}
            yield row
    yield from remaining.values()


SCHEMA = '''
CREATE TABLE observations (
 id TEXT UNIQUE NOT NULL, group_key TEXT NOT NULL, category TEXT NOT NULL,
 source TEXT NOT NULL, year INTEGER, barrio TEXT, lon REAL, lat REAL, my REAL,
 preferred INTEGER, quality REAL, score REAL, legacy INTEGER, cluster TEXT, color TEXT,
 search TEXT, nviews INTEGER DEFAULT 1
);
CREATE TABLE records (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
CREATE INDEX obs_group ON observations(group_key);
CREATE INDEX obs_category ON observations(category,id);
CREATE VIRTUAL TABLE obs_geo USING rtree(rowid,west,east,south,north);
CREATE VIRTUAL TABLE obs_fts USING fts5(search,content='observations',content_rowid='rowid',tokenize='trigram');
'''


def year(value):
    text = str(value or '')
    if not re.fullmatch(r'\d{4}-\d{2}(?:-\d{2}(?:T.*)?)?', text):
        return None
    try:
        return datetime.fromisoformat(text[:10] if len(text) >= 10 else text + '-01').year
    except ValueError:
        return None


def build(site=SITE, state=STATE):
    state.mkdir(parents=True, exist_ok=True)
    with (state / '.build.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        files = [site / p for p in ('layers/manifest.json', 'data.json', 'barrios.geojson')]
        contents = [p.read_bytes() for p in files]
        key = hashlib.sha256(b'query-schema-v5-overlap\0' + b'\0'.join(contents)).hexdigest()[:24]
        current = state / 'current.json'
        if current.exists() and json.loads(current.read_text())['query_snapshot'] == key:
            print('Query index unchanged', flush=True)
            return
        manifest, legacy, barrio_data = map(json.loads, contents)
        final = state / (key + '.sqlite3')
        temp = state / (key + '.building')
        temp.unlink(missing_ok=True)
        barrios = Barrios(barrio_data)
        db = sqlite3.connect(temp)
        try:
            db.executescript('PRAGMA journal_mode=OFF; PRAGMA synchronous=OFF;' + SCHEMA)
            for row in published_rows(site, manifest, legacy):
                row['barrio_id'], row['barrio'] = barrios.find(row['lo'], row['la'])
                search = normalize(' '.join(str(row.get(k) or '') for k in ('text', 't', 'b', 'st', 'barrio')))
                weight = (row.get('score') or 0)*math.sqrt(max(1, (row.get('w') or 1)*(row.get('h') or 1)))
                db.execute('INSERT INTO records VALUES (?,?)', (row['id'], encode(row)))
                db.execute('INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)', (
                    row['id'], row.get('duplicate_group') or row['id'], row['category'], row['source'], year(row.get('d')),
                    row['barrio_id'], row['lo'], row['la'], mercator(row['la']), int(row['id'] != row.get('group_rep')),
                    weight, row.get('score', 0), int(bool(row.get('legacy_record'))), row.get('cluster_id'), row.get('cluster_color'), search))
            db.executescript('''
                INSERT INTO obs_geo SELECT rowid,lon,lon,lat,lat FROM observations;
                INSERT INTO obs_fts(obs_fts) VALUES ('rebuild');
                CREATE TABLE objects AS
                  SELECT id,group_key,category,source,year,barrio,lon,lat,my,preferred,quality,score,legacy,cluster,color,search,n
                  FROM (SELECT *,row_number() OVER(PARTITION BY group_key ORDER BY preferred,quality DESC,id) AS rank,
                        count(*) OVER(PARTITION BY group_key) AS n FROM observations WHERE score>=.65) WHERE rank=1;
                ALTER TABLE objects RENAME COLUMN n TO nviews;
                CREATE UNIQUE INDEX object_group ON objects(group_key);
                CREATE INDEX object_category ON objects(category,id);
                UPDATE objects SET search=(SELECT group_concat(search,' ') FROM observations WHERE observations.group_key=objects.group_key AND observations.score>=.65)
                  WHERE nviews>1;
                CREATE VIRTUAL TABLE object_geo USING rtree(rowid,west,east,south,north);
                INSERT INTO object_geo SELECT rowid,lon,lon,lat,lat FROM objects;
                CREATE VIRTUAL TABLE object_fts USING fts5(search,content='objects',content_rowid='rowid',tokenize='trigram');
                INSERT INTO object_fts(object_fts) VALUES ('rebuild');
                ANALYZE;
            ''')
            counts = dict(db.execute('SELECT category,count(*) FROM observations GROUP BY category'))
            groups = dict(db.execute('SELECT category,count(*) FROM objects GROUP BY category'))
            metadata = {**manifest, 'query_snapshot': key, 'legacy': legacy['meta'], 'barrios': barrio_data.get('metadata'),
                        'min_year': db.execute('SELECT min(year) FROM observations').fetchone()[0] or 2008,
                        'max_year': db.execute('SELECT max(year) FROM observations').fetchone()[0] or datetime.now().year,
                        'categories': [{**c, 'count': counts.get(c['id'], 0), 'object_count': groups.get(c['id'], 0)} for c in manifest['categories']]}
            db.execute('CREATE TABLE metadata (json TEXT)')
            db.execute('INSERT INTO metadata VALUES (?)', (encode(metadata),))
            db.commit()
            if db.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
                raise ValueError('Query index failed integrity check')
        finally:
            db.close()
        temp.replace(final)
        pointer = current.with_suffix('.tmp')
        pointer.write_text(encode(metadata))
        pointer.replace(current)
        # Keep the preceding database for open browser tabs and in-flight exports.
        for path in sorted(state.glob('*.sqlite3'), key=lambda p: p.stat().st_mtime, reverse=True)[2:]:
            path.unlink()
        print('Published query index', key, counts, flush=True)


def bounds(value):
    if not value:
        return None
    parts = list(map(float, value.split(',')))
    if len(parts) != 4 or not all(math.isfinite(x) for x in parts):
        raise ValueError('Bounds need four finite numbers')
    w, s, e, n = parts
    if not (-180 <= w < e <= 180 and -85 <= s < n <= 85):
        raise ValueError('Invalid geographic bounds')
    return parts


def filters(params, meta):
    categories = params.get('layers', 'graffiti,street_name_plate').split(',') if params.get('layers', 'x') else []
    if set(categories) - {c['id'] for c in meta['categories']}:
        raise ValueError('Unknown category')
    source = params.get('source', '')
    if source not in ('', 'apple', 'gsv'):
        raise ValueError('Unknown source')
    lo, hi = int(params.get('from', 1900)), int(params.get('to', 2100))
    if not 1900 <= lo <= hi <= 2100:
        raise ValueError('Invalid capture year range')
    confidence = float(params.get('confidence', '.65'))
    if not math.isfinite(confidence) or not .5 <= confidence <= 1:
        raise ValueError('Confidence must be between 0.5 and 1')
    query = normalize(params.get('q', '').strip())
    if len(query) > 200:
        raise ValueError('Search is too long')
    return {'confidence': confidence, 'categories': sorted(set(categories)), 'source': source, 'from': lo, 'to': hi,
            'unknown': params.get('unknown', '1') != '0', 'grouped': params.get('group', '1') != '0',
            'all_dates': lo <= meta['min_year'] and hi >= meta['max_year'],
            'barrio': params.get('barrio', ''), 'cluster': params.get('cluster', ''), 'query': query}


def observation_where(f):
    clauses = ['category IN (' + ','.join('?' for _ in f['categories']) + ')']
    args = list(f['categories'])
    clauses.append('score>=?'); args.append(f['confidence'])
    if f['source']:
        clauses.append('source=?'); args.append(f['source'])
    clauses.append('(year BETWEEN ? AND ?' + (' OR year IS NULL)' if f['unknown'] else ')'))
    args.extend((f['from'], f['to']))
    if f['barrio']:
        clauses.append('barrio IS NULL' if f['barrio'] == 'outside' else 'barrio=?')
        if f['barrio'] != 'outside':
            args.append(f['barrio'])
    if f['cluster']:
        clauses.append("(legacy=0 OR coalesce(cluster,'')='')" if f['cluster'] == 'unassigned' else '(legacy=0 OR cluster=?)')
        if f['cluster'] != 'unassigned':
            args.append(f['cluster'])
    return ' AND '.join(clauses), args


def selection_sql(f, box=None):
    where, args = observation_where(f)
    # Groups spanning only one source/year/barrio still use the fast path when
    # no observation-level filter can remove a viewpoint.
    default = f['confidence'] == .65 and not f['source'] and not f['barrio'] and not f['cluster'] and f['unknown'] and f['all_dates']
    if not f['grouped'] or default:
        table = 'objects' if f['grouped'] else 'observations'
        index = 'object' if f['grouped'] else 'obs'
        query = f'SELECT *,{"group_key" if f["grouped"] else "id"} AS oid FROM {table} WHERE {where}'
        if f['query']:
            if len(f['query']) >= 3:
                query += f' AND rowid IN (SELECT rowid FROM {index}_fts WHERE {index}_fts MATCH ?)'
                args.append('"' + f['query'].replace('"', '""') + '"')
            query += ' AND instr(search,?)>0'; args.append(f['query'])
        if box:
            w, s, e, n = box
            # Broad city views are cheaper through the category index; use the
            # spatial index for close views that select only a few nearby streets.
            if (e-w)*(n-s) < .0001:
                query += f' AND rowid IN (SELECT rowid FROM {index}_geo WHERE west<=? AND east>=? AND south<=? AND north>=?)'
                args.extend((e, w, n, s))
            query += ' AND lon>=? AND lon<=? AND lat>=? AND lat<=?'
            args.extend((w, e, s, n))
        return query, args
    # Filter observations before choosing a representative. Bounds apply AFTER
    # grouping, so a nonrepresentative viewpoint cannot move an object into view.
    query = f'''WITH filtered AS (SELECT * FROM observations WHERE {where}),
      ranked AS (SELECT *,row_number() OVER(PARTITION BY group_key ORDER BY preferred,quality DESC,id) AS rank,
        count(*) OVER(PARTITION BY group_key) AS matching_views,
        max(instr(search,?)>0) OVER(PARTITION BY group_key) AS found FROM filtered)
      SELECT *,group_key AS oid FROM ranked WHERE rank=1 AND found=1'''
    args.append(f['query'])  # instr(text,'') is true for the unsearched case.
    if box:
        w, s, e, n = box
        query += ' AND lon>=? AND lon<=? AND lat>=? AND lat<=?'
        args.extend((w, e, s, n))
    return query, args


def connect(path):
    db = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA temp_store=MEMORY')
    return db


@lru_cache(maxsize=128)
def summary(path, serialized, serialized_box):
    f, box = json.loads(serialized), json.loads(serialized_box)
    query, args = selection_sql(f, box)
    with closing(connect(Path(path))) as db:
        rows = db.execute(f'SELECT category,count(*) AS n,min(lon) w,min(lat) s,max(lon) e,max(lat) nlat FROM ({query}) GROUP BY category', args).fetchall()
    return {'total': sum(r['n'] for r in rows), 'counts': {r['category']: r['n'] for r in rows},
            'bounds': [min(r['w'] for r in rows), min(r['s'] for r in rows), max(r['e'] for r in rows), max(r['nlat'] for r in rows)] if rows else None}


def payload(db, identity):
    return json.loads(db.execute('SELECT payload FROM records WHERE id=?', (identity,)).fetchone()[0])


def page_rows(db, f, box, limit, offset, seed=0):
    query, args = selection_sql(f, box)
    # Deterministic shuffle avoids a random sort changing pagination between calls.
    order = 'category,id' if not seed else f'((length(id)*1103515245 + unicode(substr(id,-5,1))*{seed} + unicode(substr(id,-10,1))*12345) % 2147483647),id'
    rows = db.execute(f'SELECT * FROM ({query}) ORDER BY {order} LIMIT ? OFFSET ?', [*args, limit+1, offset]).fetchall()
    items = []
    for r in rows[:limit]:
        row = payload(db, r['id'])
        keys = ('id', 'category', 'la', 'lo', 'w', 'h', 'crop', 'barrio', 'barrio_id', 'text', 't', 'cluster_color', 'legacy_record')
        item = {k: row[k] for k in keys if k in row}
        for k in ('text', 't'):
            if k in item:
                item[k] = str(item[k] or '')[:160]
        item.update(object_id=r['oid'], view_count=r['matching_views'] if 'matching_views' in r.keys() else r['nviews'])
        items.append(item)
    return items, len(rows) > limit


def object_detail(db, f, identity):
    lookup = db.execute('SELECT group_key FROM observations WHERE id=? OR group_key=? LIMIT 1', (identity, identity)).fetchone()
    if not lookup:
        raise LookupError('Object not found')
    where, args = observation_where(f)
    where += ' AND ' + ('group_key=?' if f['grouped'] else 'id=?')
    args.append(lookup['group_key'] if f['grouped'] else identity)
    rows = db.execute(f'SELECT id FROM observations WHERE {where} ORDER BY preferred,quality DESC,id', args).fetchall()
    if not rows:
        raise LookupError('No viewpoints match these filters')
    views = [payload(db, r['id']) for r in rows]
    texts = list(dict.fromkeys(str(v.get('text') or v.get('t') or '').strip() for v in views))
    return {**views[0], 'object_id': lookup['group_key'] if f['grouped'] else identity,
            'views': views, 'view_count': len(views), 'group_text': '\n'.join(t for t in texts if t)}


def cells(db, f, box, cols, rows, meta, cluster_colors=False):
    query, args = selection_sql(f, box)
    w, s, e, n = box
    bottom, top = mercator(s), mercator(n)
    dx, dy = (e-w)/cols, (top-bottom)/rows
    shade = "CASE WHEN legacy THEN coalesce(color,'#777') ELSE '' END" if cluster_colors else "''"
    sql = f'''SELECT min(?,max(0,cast((lon-?)/? AS INTEGER))) cx,
      min(?,max(0,cast((my-?)/? AS INTEGER))) cy,
      category,{shade} shade,
      count(*) amount,avg(lon) lon,avg(my) my,min(oid) identity
      FROM ({query}) GROUP BY cx,cy,category,shade'''
    found = db.execute(sql, [cols-1, w, dx, rows-1, bottom, dy, *args]).fetchall()
    colors = {c['id']: c['color'] for c in meta['categories']}
    groups = {}
    for r in found:
        key = (r['cx'], r['cy'])
        group = groups.setdefault(key, {'lo': 0, 'my': 0, 'count': 0, 'colors': [], 'id': r['identity'],
            'bounds': [w+r['cx']*dx, latitude(bottom+r['cy']*dy), w+(r['cx']+1)*dx, latitude(bottom+(r['cy']+1)*dy)]})
        group['lo'] += r['lon']*r['amount']; group['my'] += r['my']*r['amount']; group['count'] += r['amount']
        shade = r['shade'] or colors[r['category']]
        if shade not in group['colors']:
            group['colors'].append(shade)
    for group in groups.values():
        group['lo'] /= group['count']; group['la'] = latitude(group.pop('my')/group['count'])
        if group['count'] != 1:
            group.pop('id')
    return list(groups.values())


def export_record(row, origin, crop_base):
    views = row['views']
    def crop(url):
        return (crop_base or origin).rstrip('/') + '/' + url.lstrip('/') if url else None
    def context(url):
        return origin.rstrip('/') + '/' + url.lstrip('/') if url else None
    return {
        'id': row['object_id'], 'observation_id': row['id'], 'group_id': row.get('duplicate_group'),
        'member_ids': [v['id'] for v in views], 'observation_count': len(views), 'category': row['category'],
        'category_status': row.get('status'), 'source': row['source'], 'sources': list(dict.fromkeys(v['source'] for v in views)),
        'capture_date': row.get('d'), 'capture_dates': list(dict.fromkeys(v.get('d') for v in views if v.get('d'))),
        'latitude': row['la'], 'longitude': row['lo'], 'location_basis': row.get('location_basis'),
        'barrio': row.get('barrio'), 'barrio_id': row.get('barrio_id'), 'barrio_basis': 'recorded_position_in_municipal_polygon',
        'score': row.get('score'), 'prompt': row.get('prompt'), 'text': row.get('group_text') or row.get('text') or row.get('t') or '',
        'ocr_model': row.get('ocr_model'), 'view_id': row.get('view_id'), 'run_id': row.get('run_id'),
        'snapshot_id': row.get('snapshot_id'), 'provider_url': row.get('u'), 'crop_url': crop(row.get('crop')),
        'context_url': context(row.get('raw')), 'box_normalized': row.get('bx'),
        'observations': [{'id': v['id'], 'source': v['source'], 'capture_date': v.get('d'), 'latitude': v['la'], 'longitude': v['lo'],
                          'location_basis': v.get('location_basis'), 'barrio': v.get('barrio'), 'barrio_id': v.get('barrio_id'),
                          'score': v.get('score'), 'text': v.get('text') or v.get('t') or '', 'run_id': v.get('run_id'),
                          'view_id': v.get('view_id'), 'provider_url': v.get('u'), 'crop_url': crop(v.get('crop')), 'context_url': context(v.get('raw'))} for v in views],
    }


def csv_cell(value):
    text = '' if value is None else encode(value) if isinstance(value, (dict, list)) else str(value)
    if isinstance(value, str) and text.lstrip().startswith(('=', '+', '-', '@')):
        text = "'" + text
    return '"' + text.replace('"', '""') + '"'


class API:
    def __init__(self, state=STATE, site=SITE, origin=None):
        self.state, self.site, self.origin = state, site, origin or os.environ.get('TYPOLOGY_PUBLIC_URL', 'http://127.0.0.1:8790')
        self.capacity = threading.BoundedSemaphore(4)

    def metadata(self):
        return json.loads((self.state / 'current.json').read_text())

    def database(self, params):
        key = params.get('snapshot') or self.metadata()['query_snapshot']
        if not re.fullmatch('[a-f0-9]{24}', key):
            raise ValueError('Invalid query snapshot')
        path = self.state / (key + '.sqlite3')
        if path.exists():
            with sqlite3.connect(path.as_uri() + '?mode=ro', uri=True) as db:
                if 'score' not in {r[1] for r in db.execute('PRAGMA table_info(observations)')}:
                    raise FileNotFoundError('Snapshot expired; update the page')
        if not path.exists():
            raise FileNotFoundError('Snapshot expired; update the page')
        return path

    def query(self, path, db, meta, params):
        f = filters(params, meta)
        box = bounds(params.get('bbox'))
        area = box if params.get('area') == '1' else None
        result = dict(summary(str(path), encode(f), encode(area)))
        limit, offset = min(120, max(1, int(params.get('limit', 48)))), max(0, int(params.get('offset', 0)))
        if offset > 2000000:
            raise ValueError('Offset too large')
        page_box = bounds(params.get('list_bbox')) or (box if params.get('view', 'map') == 'map' else area)
        seed = int(params.get('seed', 0)) % 2147483647
        result['items'], result['has_more'] = page_rows(db, f, page_box, limit, offset, seed)
        result['offset'] = offset
        if params.get('map', '1') == '1' and box:
            cols, rows = (min(128, max(1, int(params.get(k, 64)))) for k in ('cols', 'rows'))
            result['cells'] = cells(db, f, box, cols, rows, meta, params.get('colors') == '1')
            result['visible'] = sum(c['count'] for c in result['cells'])
        result['snapshot'] = meta['query_snapshot']
        return result

    def handler(self):
        api = self
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if not api.capacity.acquire(timeout=10):
                    self.send_error(503, 'Busy; retry shortly'); return
                try:
                    route = urlsplit(self.path)
                    params = {k: v[-1] for k, v in parse_qs(route.query, keep_blank_values=True).items()}
                    if route.path == '/api/meta':
                        self.json(api.metadata()); return
                    path = api.database(params)
                    with closing(connect(path)) as db:
                        meta = json.loads(db.execute('SELECT json FROM metadata').fetchone()[0])
                        if route.path == '/api/query':
                            self.json(api.query(path, db, meta, params))
                        elif route.path == '/api/object':
                            self.json(object_detail(db, filters(params, meta), params.get('id', '')))
                        elif route.path == '/api/export':
                            self.export(path, db, meta, params)
                        else:
                            self.send_error(404)
                except FileNotFoundError as error:
                    self.send_error(409, str(error))
                except LookupError as error:
                    self.send_error(404, str(error))
                except (ValueError, OverflowError) as error:
                    self.send_error(400, str(error))
                except (BrokenPipeError, ConnectionResetError):
                    pass
                except Exception as error:
                    self.log_error('%s', type(error).__name__)
                    self.send_error(500, 'Query failed')
                finally:
                    api.capacity.release()

            def json(self, value):
                body = encode(value).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers(); self.wfile.write(body)

            def export(self, path, db, meta, params):
                f = filters(params, meta)
                box = bounds(params.get('bbox')) if params.get('area') == '1' else None
                fmt = params.get('format', 'jsonl')
                if fmt not in ('jsonl', 'csv', 'geojson', 'manifest'):
                    raise ValueError('Unknown download format')
                totals = summary(str(path), encode(f), encode(box))
                metadata = {'schema_version': 1, 'exported_at': datetime.now(timezone.utc).isoformat(),
                            'city': meta.get('city'), 'record_type': 'object_groups' if f['grouped'] else 'observations',
                            'record_count': totals['total'], 'filters': f, 'bounds': box, 'snapshot': meta,
                            'selection': 'Observation filters precede grouping; bounds apply to the retained representative. All matching viewpoints are included.'}
                query, args = selection_sql(f, box)
                crop_file = api.site / 'crop-storage.json'
                crop_base = json.loads(crop_file.read_text()).get('base_url') if crop_file.exists() else None
                self.send_response(200)
                self.send_header('Content-Type', {'csv': 'text/csv; charset=utf-8', 'jsonl': 'application/x-ndjson', 'geojson': 'application/geo+json', 'manifest': 'application/json'}[fmt])
                extension = 'json' if fmt == 'manifest' else fmt
                self.send_header('Content-Disposition', f'attachment; filename="typology-coruna-{fmt}.{extension}"')
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                def write(text):
                    self.wfile.write(text.encode())
                if fmt == 'manifest':
                    write(encode(metadata)); return
                if fmt == 'geojson':
                    write('{"type":"FeatureCollection","metadata":' + encode(metadata) + ',"features":[')
                fields = None
                for index, r in enumerate(db.execute(f'SELECT * FROM ({query}) ORDER BY category,id', args)):
                    nviews = r['matching_views'] if 'matching_views' in r.keys() else r['nviews']
                    if nviews == 1:
                        row = payload(db, r['id']); row = {**row, 'object_id': r['oid'], 'views': [row]}
                    else:
                        row = object_detail(db, f, r['oid'])
                    record = export_record(row, api.origin, crop_base)
                    if fmt == 'jsonl':
                        write(encode(record) + '\n')
                    elif fmt == 'csv':
                        if fields is None:
                            fields = list(record); write('\ufeff' + ','.join(csv_cell(k) for k in fields) + '\r\n')
                        write(','.join(csv_cell(record[k]) for k in fields) + '\r\n')
                    else:
                        write((',' if index else '') + encode({'type': 'Feature', 'id': record['id'], 'geometry': {'type': 'Point', 'coordinates': [record['longitude'], record['latitude']]}, 'properties': record}))
                if fmt == 'geojson':
                    write(']}')
        return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['build', 'serve'])
    parser.add_argument('--site', type=Path, default=SITE)
    parser.add_argument('--state', type=Path, default=STATE)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8794)
    parser.add_argument('--origin', default=os.environ.get('TYPOLOGY_PUBLIC_URL', 'http://127.0.0.1:8790'), help='Public viewer URL used in exports')
    args = parser.parse_args()
    if args.command == 'build':
        build(args.site, args.state)
    else:
        ThreadingHTTPServer((args.host, args.port), API(args.state, args.site, args.origin).handler()).serve_forever()


if __name__ == '__main__':
    main()
