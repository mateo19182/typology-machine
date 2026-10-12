"""Explicitly selected duplicate proposals used by map exports; never infer merges."""
import hashlib
import json


def load_groups(city_base, run_id, source):
    candidates = [
        (city_base / source / 'store_sign/strict_dedupe_v13', 'store_sign'),
        (city_base / source / 'multicategory_dedupe/pilot_v3', None),
    ]
    membership, sources = {}, []
    for folder, category in candidates:
        path = folder / 'groups.jsonl'
        if not path.exists():
            continue
        config = json.loads((folder / 'config.json').read_text())
        if category is None and config.get('run_id') != run_id:
            continue
        data = path.read_bytes()
        sources.append({'path': str(folder.relative_to(city_base)), 'sha256': hashlib.sha256(data).hexdigest()})
        for line in data.splitlines():
            group = json.loads(line)
            if len(group['members']) < 2:
                continue
            cat = category or group['category']
            gid = f"{source}:{cat}:{group.get('sign', group.get('id'))}"
            def crop_id(value):
                return f'legacy:{source}:{value}' if category else value
            rep = crop_id(group['rep'])
            for value in group['members']:
                key = crop_id(value)
                if key in membership:
                    raise ValueError(f'Overlapping map duplicate groups: {key}')
                membership[key] = {'duplicate_group': gid, 'group_rep': rep, 'group_category': cat}
    return membership, sources
