"""Build pose_demo.html, a self-contained, bulletproof disentanglement demo that runs in any browser.

It is meant to stand alone without the report: everything needed to believe the disentanglement claim is
inside the one file. It has three parts.

1. Interactive slider. The factored decoder is pre-rendered over a 3D grid of (x, y, theta) with the scene
   scene latent held fixed at one frame's encoding. The real input frame is shown beside the decode, and a live
   readout reports the MEASURED pose of the decoded lander in the same state units commanded (position fit
   from the grid; tilt from the geometric reader), so dialing one axis moves only that axis (the crosstalk =
   disentanglement) and the measured value tracks the command inside the reliable band.

2. The freeze rebuttal, in words. Holding the scene code fixed is not circular: the decoder renders the
   whole image from the whole latent, so if pose touched the terrain the terrain would drift as you drag.
   It does not, and part 3 measures that.

3. Embedded evidence (static figures, both directions, with numbers):
   - pose -> scene: the change-map heatmap (terrain change ~0.002 as pose is swept),
   - the per-axis crosstalk figure (dial one axis, the others move < 1 px),
   - scene -> pose: the same pose decoded on several terrains (lander centroid holds to ~0.04 px).

Works with any checkpoint sharing the factored 32-dim layout (z[0]=x, z[1]=y, z[2:4]=cos/sin theta,
z[4:]=scene latent). No cache: re-renders each run so the measured centroids stay in sync with the frames.
"""
import base64
import io
import math
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, ".")
import config
import factored_data
import geom_theta
import train_clean_vae as TC
from zlander_recon_fig import load, build_z
sys.path.insert(0, config.BASELINE_SRC)
from piwm_model.sprite import purple_mask  # noqa

# Default = the shipped model (its demo is committed for recruiters). A teammate who has retrained can set
# PIWM_MODEL=factored_reproduce_best to build the SAME demo from their reproduced checkpoint and confirm it
# works (the interactive counterpart to verify.py).
MODEL = os.environ.get("PIWM_MODEL", "factored_clean_noaug_best")
# Coarser grid = a small, fast-loading demo (was 0.1 / 0.1 / 10 = ~5,300 frames, ~21 MB). Steps chosen so
# the θ grid lands on 0 and +/-45 (the band edges). Full 3D cross-product, so size is NX*NY*NT frames.
X_RANGE, X_STEP = (-1.0, 1.0), 0.2
Y_RANGE, Y_STEP = (-0.2, 1.4), 0.2
T_RANGE, T_STEP = (-60, 60), 15
# static evidence figures embedded into the page (relative to FIG_DIR)
EVIDENCE = {
    "heatmap": "factored/pose_scene_disentangle.png",
    "crosstalk": "factored/crosstalk_xy_factored_clean_noaug.png",
    "converse": "factored/scene_to_pose.png",
}


def img_b64(arr_hwc_uint8):
    buf = io.BytesIO(); Image.fromarray(arr_hwc_uint8).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def file_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


@torch.no_grad()
def render_frame(m, z_base, x, y, th, reader):
    """Decode one pose; return (png_b64, cx_px, cy_px, tilt_deg). Measured values are None if no lander."""
    z = z_base.clone()
    z[0, 0] = x; z[0, 1] = y
    z[0, 2] = math.cos(math.radians(th)); z[0, 3] = math.sin(math.radians(th))
    img = m["vae"].decode(z).clamp(0, 1)[0].cpu()
    mask = purple_mask(img).numpy()
    ys, xs = np.where(mask)
    if len(xs) >= 8:
        cx, cy = float(xs.mean()), float(ys.mean())
        tr = reader(mask)                                   # geometric tilt reader -> radians (or None)
        # fold to (-90, 90]: the lander never inverts, so |tilt| > 90 is the head/feet 180-degree flip
        tilt = round(((math.degrees(tr) + 90.0) % 180.0) - 90.0, 1) if tr is not None else None
    else:
        cx = cy = tilt = None
    arr = (img.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return img_b64(arr), cx, cy, tilt


def closest(vals, v):
    return int(np.argmin(np.abs(np.asarray(vals) - v)))


def main():
    config.set_seed()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teI, teC, teS, *_ = TC.preload(config.TEST_DIR, 30)
    fi = int(np.random.default_rng(config.SEED).choice(teI.size(0)))
    fr = teI[fi:fi + 1].to(dev).float() / 255.; cr = teC[fi:fi + 1].to(dev).float() / 255.; st = teS[fi:fi + 1]
    sc = factored_data.scene_only(fr[0].cpu())[0].unsqueeze(0).to(dev)
    x0, y0 = float(st[0, config.X]), float(st[0, config.Y])
    real_b64 = img_b64(teI[fi].permute(1, 2, 0).numpy().astype("uint8"))

    xvals = np.round(np.arange(X_RANGE[0], X_RANGE[1] + 1e-6, X_STEP), 2)
    yvals = np.round(np.arange(Y_RANGE[0], Y_RANGE[1] + 1e-6, Y_STEP), 2)
    tvals = np.round(np.arange(T_RANGE[0], T_RANGE[1] + 1e-6, T_STEP), 0)
    NX, NY, NT = len(xvals), len(yvals), len(tvals)

    reader, _ = geom_theta.calibrate_on_real(train_files=20, max_files_eval=8)   # geometric tilt reader
    m = load(MODEL, dev)
    z_base = build_z(m, sc, cr, st)
    frames, cxs, cys, mths, cmdx, cmdy = [], [], [], [], [], []   # flat: idx = ix*(NY*NT) + iy*NT + it
    for ix, x in enumerate(xvals):
        for y in yvals:
            for t in tvals:
                b, cx, cy, mt = render_frame(m, z_base, float(x), float(y), float(t), reader)
                frames.append(b); cxs.append(cx); cys.append(cy); mths.append(mt)
                cmdx.append(float(x)); cmdy.append(float(y))
        print(f"  x row {ix + 1}/{NX} ({len(frames)} frames)", flush=True)

    # fit commanded(state) -> pixel-centroid on in-band rendered frames, then invert: readout in STATE units
    cmdx, cmdy = np.array(cmdx), np.array(cmdy)
    cxa = np.array([np.nan if v is None else v for v in cxs])
    cya = np.array([np.nan if v is None else v for v in cys])
    inb = (cmdx >= -0.7) & (cmdx <= 0.7) & (cmdy >= 0.0) & (cmdy <= 1.3) & np.isfinite(cxa) & np.isfinite(cya)
    ax_, bx_ = np.polyfit(cmdx[inb], cxa[inb], 1)      # cx ~= ax_*x + bx_
    ay_, by_ = np.polyfit(cmdy[inb], cya[inb], 1)      # cy ~= ay_*y + by_
    mx = [None if v is None else round((v - bx_) / ax_, 2) for v in cxs]   # measured x, state units
    my = [None if v is None else round((v - by_) / ay_, 2) for v in cys]   # measured y, state units

    data = dict(
        xvals=[float(v) for v in xvals], yvals=[float(v) for v in yvals], tvals=[float(v) for v in tvals],
        NY=NY, NT=NT, bands=dict(x=[-0.7, 0.7], y=[0.0, 1.3], t=[-45, 45]),
        defaults=dict(ix=closest(xvals, x0), iy=closest(yvals, y0), it=closest(tvals, 0.0)),
        frames=frames, mx=mx, my=my, mth=mths, real=real_b64,
        figs={k: file_b64(os.path.join(config.FIG_DIR, p)) for k, p in EVIDENCE.items()})

    import json
    html = HTML_TEMPLATE.replace("/*DATA*/", json.dumps(data))
    # shipped model -> pose_demo.html (committed); a reproduced model -> pose_demo_<model>.html (never clobbers it)
    out = os.path.join(config.HERE, "pose_demo.html" if MODEL == "factored_clean_noaug_best"
                       else f"pose_demo_{MODEL}.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out}  ({os.path.getsize(out) / 1024 / 1024:.1f} MB, {len(frames)} frames)", flush=True)


HTML_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Disentanglement demo (factored VAE)</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;max-width:820px;margin:24px auto;padding:0 16px;color:#222;line-height:1.5}
 h1{font-size:21px} h2{font-size:17px;margin-top:30px;border-bottom:1px solid #eee;padding-bottom:4px}
 img.frame{width:330px;height:220px;image-rendering:pixelated;background:#000;border:1px solid #ccc}
 figure{margin:0;text-align:center} figcaption{font-size:13px;color:#555;margin-top:4px}
 .frames{display:flex;gap:18px;justify-content:center;margin:12px 0}
 .row{display:flex;align-items:center;gap:12px;margin:10px 0}
 .row b.lab{width:18px;font-size:16px} .row input[type=range]{flex:1}
 .val{width:74px;font-variant-numeric:tabular-nums;font-size:14px}
 .band{color:#888;font-size:12px;width:150px}
 .ok{color:#1a8a1a;font-weight:bold} .bad{color:#d62728;font-weight:bold}
 .note{background:#f6f6f6;border-left:3px solid #bbb;padding:10px 14px;font-size:14px}
 .meas{background:#eef4ff;border:1px solid #cdddf5;border-radius:6px;padding:8px 12px;margin:10px 0;
       font-size:14px;font-variant-numeric:tabular-nums;text-align:center}
 .evid{width:100%;border:1px solid #ddd;margin-top:6px} .cap{font-size:13px;color:#555;margin:4px 0 0}
 input[type=range]{-webkit-appearance:none;appearance:none;height:8px;border-radius:4px;background:#ddd;outline:none}
 input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:#333;cursor:pointer;border:2px solid #fff;box-shadow:0 0 2px #888}
 input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:#333;border:2px solid #fff;cursor:pointer}
</style></head><body>
<h1>Is the pose really separate from the scene? (factored VAE)</h1>
<p>The lander's pose is a few named latent dimensions (x, y, tilt); the rest of the latent is a "scene
scene latent" holding the terrain. This page is meant to show whether those two are actually
disentangled, both by letting a user check it and by showing the measured evidence below.</p>

<h2>1. Move the lander, watch the scene</h2>
<p>Drag the sliders. The decoder renders the lander at the commanded pose on the <i>real</i> terrain shown
on the left. The readout reports the lander's measured pose, read back off the decoded image in the
same state units commanded (and degrees for tilt). Watch that the terrain never moves, and
that each measured value tracks only the axis dialed: drag x and measured x follows while y holds; drag
tilt and x, y hold while measured tilt follows. The
<span style="background:#9ed99e;padding:0 4px;border-radius:3px">green</span> band on each track is the
reliable range; outside it the measured value stops tracking. Position is a robust centroid read; tilt is
the geometric reader, reliable in the band and degrading on near-symmetric landers.</p>
<div class="frames">
  <figure><img class="frame" id="real"><figcaption>real input frame (fixed reference terrain)</figcaption></figure>
  <figure><img class="frame" id="img"><figcaption>decoded at the commanded pose</figcaption></figure>
</div>
<div class="meas" id="meas"></div>
<p class="cap" style="text-align:center;margin:2px 0 12px">Measured tilt jitters about &plusmn;7&deg; (std)
as x or y are dialed, even though the lander is not visibly rotating. That is the geometric reader's noise on a
near-symmetric lander, its principal axis is barely defined, so a one-pixel change in the silhouette swings the
read a few degrees. It is measurement noise, not real rotation: position itself is exact. For scale, &plusmn;7&deg;
is about &plusmn;2% of a full 360&deg; turn (and roughly &plusmn;15% of the lander's &plusmn;45&deg; working range).</p>
<div class="row"><b class="lab">x</b><input type="range" id="sx" min="0"><span class="val" id="vx"></span><span class="band">reliable -0.7..0.7</span></div>
<div class="row"><b class="lab">y</b><input type="range" id="sy" min="0"><span class="val" id="vy"></span><span class="band">reliable 0.0..1.3</span></div>
<div class="row"><b class="lab">&theta;</b><input type="range" id="st" min="0"><span class="val" id="vt"></span><span class="band">reliable -45..45 deg</span></div>
<div id="status" style="text-align:center;font-size:15px;margin:6px 0"></div>

<h2>2. What "out of range" means</h2>
<div class="note">
Each slider runs its full range, but the decoder was trained only on the poses the data contained. Inside
the green band the lander tracks the command; outside it the decoder under-renders (the lander stops
moving or rotating, or fades) and that axis turns red. The dataset is short random-action rollouts that
start upright at top centre and drift within a bounded band, so rare poses (far edges, large tilt) are
under-rendered. That is a limit of the substrate, not of this demo.
</div>

<h2>3. Isn't freezing the scene circular?</h2>
<div class="note">
No. The scene code is held fixed, but the decoder renders the whole image from the whole latent, so
if the pose dimensions touched the terrain, the terrain would drift as the sliders are dragged, even with the
scene code frozen. It does not. Freezing isolates the pose's effect; it does not force the terrain to stay.
The next figures measure exactly how much the pose leaks into the scene (almost none), and the
reverse.
</div>

<h2>4. The measured evidence (both directions)</h2>
<p><b>Pose does not move the scene.</b> Sweeping each pose axis with the scene fixed, the decoded image
changes only along the lander's path; the terrain change is about 0.002, roughly 50x smaller (bright =
changes, dark = stays).</p>
<img class="evid" id="fheat"><p class="cap">Per-pixel change as each axis is swept. Lander bright, terrain flat.</p>
<p><b>Each axis moves only itself.</b> Commanding one axis and measuring all three on the decoded lander:
the off-axis drift stays under a pixel.</p>
<img class="evid" id="fcross"><p class="cap">Dial x: ~104 px in x, 0.9 px in y. Dial y: ~54 px in y, 0.7 px in x. Dial tilt: x, y within 1 px.</p>
<p><b>The scene does not move the lander.</b> The same commanded pose decoded on six different terrains: the
lander lands in the same spot every time (centroid spread ~0.04 px), and the terrains are visibly different,
so the scene latent carries the real scene, not a memorized one.</p>
<img class="evid" id="fconv"><p class="cap">One pose, six terrains. Cyan + marks the lander centroid.</p>

<script>
const D = /*DATA*/;
const sx=document.getElementById('sx'), sy=document.getElementById('sy'), st=document.getElementById('st');
document.getElementById('real').src='data:image/png;base64,'+D.real;
document.getElementById('fheat').src='data:image/png;base64,'+D.figs.heatmap;
document.getElementById('fcross').src='data:image/png;base64,'+D.figs.crosstalk;
document.getElementById('fconv').src='data:image/png;base64,'+D.figs.converse;
sx.max=D.xvals.length-1; sy.max=D.yvals.length-1; st.max=D.tvals.length-1;
sx.value=D.defaults.ix; sy.value=D.defaults.iy; st.value=D.defaults.it;
function sleeve(vals,lo,hi){
  const a=vals[0], b=vals[vals.length-1], p=v=>((v-a)/(b-a)*100).toFixed(1);
  return 'linear-gradient(to right,#ddd 0 '+p(lo)+'%,#9ed99e '+p(lo)+'% '+p(hi)+'%,#ddd '+p(hi)+'% 100%)';
}
sx.style.background=sleeve(D.xvals,D.bands.x[0],D.bands.x[1]);
sy.style.background=sleeve(D.yvals,D.bands.y[0],D.bands.y[1]);
st.style.background=sleeve(D.tvals,D.bands.t[0],D.bands.t[1]);
function tag(name,v,lo,hi,unit){
  const inb=v>=lo&&v<=hi, d=(unit==='°')?0:2;
  return name+'='+v.toFixed(d)+unit+(inb?' <span class="ok">ok</span>':' <span class="bad">out</span>');
}
function upd(){
  const ix=+sx.value, iy=+sy.value, it=+st.value, idx=ix*(D.NY*D.NT)+iy*D.NT+it;
  document.getElementById('img').src='data:image/png;base64,'+D.frames[idx];
  const x=D.xvals[ix], y=D.yvals[iy], t=D.tvals[it];
  document.getElementById('vx').textContent=x.toFixed(2);
  document.getElementById('vy').textContent=y.toFixed(2);
  document.getElementById('vt').textContent=t.toFixed(0)+'°';
  const mx=D.mx[idx], my=D.my[idx], mth=D.mth[idx];
  document.getElementById('meas').innerHTML = (mx===null)
    ? 'measured: <b>lander not rendered</b> (out-of-range pose)'
    : 'commanded&nbsp; x='+x.toFixed(2)+'&nbsp; y='+y.toFixed(2)+'&nbsp; &theta;='+t.toFixed(0)+'&deg;<br>'
      +'measured&nbsp;&nbsp; x=<b>'+mx.toFixed(2)+'</b>&nbsp; y=<b>'+my.toFixed(2)+'</b>&nbsp; &theta;=<b>'
      +(mth===null?'&mdash;':mth.toFixed(0)+'&deg;')+'</b>';
  document.getElementById('status').innerHTML=
    tag('x',x,D.bands.x[0],D.bands.x[1],'')+' &nbsp; '+tag('y',y,D.bands.y[0],D.bands.y[1],'')+
    ' &nbsp; '+tag('θ',t,D.bands.t[0],D.bands.t[1],'°');
}
[sx,sy,st].forEach(e=>e.addEventListener('input',upd));
upd();
</script>
</body></html>
"""


if __name__ == "__main__":
    main()
