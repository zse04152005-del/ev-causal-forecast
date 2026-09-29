# minimal shapefile (polygon) reader + coordinate helpers, no third-party GIS deps
import struct, numpy as np, math
R = 6378137.0
def read_dbf(path):
    f = open(path, "rb").read(); n = struct.unpack("<I", f[4:8])[0]; hl = struct.unpack("<H", f[8:10])[0]; rl = struct.unpack("<H", f[10:12])[0]
    fields = []; i = 32
    while f[i] != 0x0D:
        fields.append((f[i:i+11].split(b"\x00")[0].decode(), chr(f[i+11]), f[i+16])); i += 32
    recs = []
    for r in range(n):
        p = hl + r * rl + 1; row = {}
        for name, typ, ln in fields:
            v = f[p:p+ln].decode("utf-8", "replace").strip(); p += ln
            row[name] = float(v) if typ in "NF" and v else v
        recs.append(row)
    return recs
def read_shp(path):
    f = open(path, "rb").read(); pos = 100; shapes = []
    while pos < len(f):
        clen = struct.unpack(">i", f[pos+4:pos+8])[0] * 2; c = f[pos+8:pos+8+clen]; pos += 8 + clen
        st = struct.unpack("<i", c[:4])[0]
        if st == 0: shapes.append([]); continue
        nparts, npts = struct.unpack("<2i", c[36:44]); parts = list(struct.unpack(f"<{nparts}i", c[44:44+4*nparts]))
        pts = np.frombuffer(c[44+4*nparts:44+4*nparts+16*npts], dtype="<f8").reshape(npts, 2)
        parts.append(npts); shapes.append([pts[parts[k]:parts[k+1]] for k in range(nparts)])
    return shapes
def merc_to_ll(x, y):
    return np.degrees(x / R), np.degrees(2 * np.arctan(np.exp(y / R)) - np.pi / 2)
def ll_to_merc(lon, lat):
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))
_a = 6378245.0; _ee = 0.00669342162296594323
def _tlat(x, y):
    r = -100 + 2*x + 3*y + 0.2*y*y + 0.1*x*y + 0.2*np.sqrt(np.abs(x))
    r += (20*np.sin(6*x*np.pi) + 20*np.sin(2*x*np.pi)) * 2/3; r += (20*np.sin(y*np.pi) + 40*np.sin(y/3*np.pi)) * 2/3
    r += (160*np.sin(y/12*np.pi) + 320*np.sin(y*np.pi/30)) * 2/3; return r
def _tlon(x, y):
    r = 300 + x + 2*y + 0.1*x*x + 0.1*x*y + 0.1*np.sqrt(np.abs(x))
    r += (20*np.sin(6*x*np.pi) + 20*np.sin(2*x*np.pi)) * 2/3; r += (20*np.sin(x*np.pi) + 40*np.sin(x/3*np.pi)) * 2/3
    r += (150*np.sin(x/12*np.pi) + 300*np.sin(x/30*np.pi)) * 2/3; return r
def wgs_to_gcj(lon, lat):
    dlat = _tlat(lon - 105, lat - 35); dlon = _tlon(lon - 105, lat - 35); rl = np.radians(lat)
    m = 1 - _ee * np.sin(rl) ** 2; sm = np.sqrt(m)
    dlat = dlat * 180 / ((_a * (1 - _ee)) / (m * sm) * np.pi); dlon = dlon * 180 / (_a / sm * np.cos(rl) * np.pi)
    return lon + dlon, lat + dlat
def gcj_to_wgs(lon, lat):
    wl, wa = np.array(lon, float), np.array(lat, float)
    for _ in range(4):
        gl, ga = wgs_to_gcj(wl, wa); wl, wa = wl - (gl - lon), wa - (ga - lat)
    return wl, wa
def assign_points(shapes, x, y):
    """even-odd point-in-polygon over all parts; returns polygon index or -1"""
    from matplotlib.path import Path
    out = np.full(len(x), -1, int)
    for k, parts in enumerate(shapes):
        allp = np.vstack(parts); x0, y0 = allp.min(0); x1, y1 = allp.max(0)
        cand = np.where((x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (out < 0))[0]
        if len(cand) == 0: continue
        pts = np.c_[x[cand], y[cand]]; cnt = np.zeros(len(cand), int)
        for ring in parts: cnt += Path(ring).contains_points(pts)
        out[cand[cnt % 2 == 1]] = k
    return out
def ring_area(r):
    x, y = r[:, 0], r[:, 1]; return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
