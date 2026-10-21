"""Write a time-varying version of each calibrated route file.

The calibrated demand is constant for the whole hour and symmetric, which is the case a
fixed-time plan is designed for (Webster 1958). Real demand changes during the hour. This
keeps the calibrated OD pattern and the hourly mean, and splits the hour into four 900 s
periods with different multipliers for the north-south and east-west streams:

    period        0-900   900-1800   1800-2700   2700-3600   mean
    north-south    1.3      1.1         0.7         0.9       1.0
    east-west      0.7      1.1         1.3         0.9       1.0

A flow belongs to the stream of the edge it enters on (north_*/south_* or east_*/west_*).

    python tools/make_dynamic_routes.py            # writes sumo_files/<net>/dynamic.rou.xml
"""

import copy
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PERIODS = [(0, 900, 1.3, 0.7), (900, 1800, 1.1, 1.1), (1800, 2700, 0.7, 1.3), (2700, 3600, 0.9, 0.9)]
NETS = ['single', 'grid4', 'grid8']


def entry_edge(flow, routes):
    if flow.get('route'):
        return routes[flow.get('route')][0]
    return flow.get('from')


def stream(edge):
    if edge.startswith(('north', 'south')):
        return 'ns'
    if edge.startswith(('east', 'west')):
        return 'ew'
    raise ValueError(f'cannot tell the stream of entry edge {edge!r}')


def write_dynamic(net):
    src = ROOT / 'sumo_files' / net / 'calibrated.rou.xml'
    tree = ET.parse(src)
    root = tree.getroot()
    routes = {r.get('id'): r.get('edges').split() for r in root.iter('route') if r.get('id')}
    flows = [f for f in root if f.tag == 'flow']
    for f in flows:
        assert (f.get('begin'), f.get('end')) == ('0', '3600'), f'{net}: {f.get("id")} is not a 0-3600 s flow'
        s = stream(entry_edge(f, routes))
        pos = list(root).index(f)
        root.remove(f)
        for i, (b, e, m_ns, m_ew) in enumerate(PERIODS):
            g = copy.deepcopy(f)
            p = float(f.get('probability')) * (m_ns if s == 'ns' else m_ew)
            assert p <= 1.0
            g.set('id', f'{f.get("id")}_p{i + 1}')
            g.set('begin', str(b))
            g.set('end', str(e))
            g.set('probability', f'{p:.5f}')
            root.insert(pos + i, g)
    # SUMO wants flows sorted by begin time
    flows = sorted([f for f in root if f.tag == 'flow'], key=lambda f: float(f.get('begin')))
    for f in flows:
        root.remove(f)
    root.extend(flows)
    out = src.with_name('dynamic.rou.xml')
    tree.write(out, encoding='UTF-8', xml_declaration=True)
    print(f'{out.relative_to(ROOT)}: {len(flows)} flows')


if __name__ == '__main__':
    for net in NETS:
        write_dynamic(net)
