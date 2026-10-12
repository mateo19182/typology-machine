"""Fetch Coruña's municipal Barrios / AA VV polygons for the map's barrio filter."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import requests

SOURCE = 'https://ide.coruna.gal/server/rest/services/Publica/CB_DIVISIONES_ADMINISTRATIVAS/MapServer/3'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('site/barrios.geojson'))
    args = parser.parse_args()
    response = requests.get(SOURCE + '/query', params={
        'where': '1=1', 'outFields': 'objectid,nom_barrio', 'outSR': 4326,
        'returnGeometry': 'true', 'f': 'geojson', 'orderByFields': 'objectid',
    }, timeout=60)
    response.raise_for_status()
    data = response.json()
    if data.get('type') != 'FeatureCollection' or not data.get('features'):
        raise ValueError('Municipal service returned no barrio polygons')
    if data.get('exceededTransferLimit') or data.get('properties', {}).get('exceededTransferLimit'):
        raise ValueError('Municipal service truncated the barrio inventory')
    for feature in data['features']:
        props = feature['properties']
        if feature['geometry']['type'] not in ('Polygon', 'MultiPolygon'):
            raise ValueError('Expected polygon geometry')
        feature['id'] = str(props['objectid'])
        feature['properties'] = {'id': str(props['objectid']), 'name': props['nom_barrio']}
    data['metadata'] = {
        'source': SOURCE, 'layer': 'Barrios / AA VV',
        'attribution': 'Concello da Coruña / Ayuntamiento da Coruña',
        'retrieved_at': datetime.now(timezone.utc).isoformat(),
        'assignment': 'Point within municipal polygon; observation position may be the camera position.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')) + '\n')
    temporary.replace(args.out)
    print(f"Saved {len(data['features'])} barrio polygons to {args.out}")


if __name__ == '__main__':
    main()
