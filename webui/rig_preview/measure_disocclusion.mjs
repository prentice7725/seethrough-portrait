// Measures what a head turn exposes, for H1 and H6.
//
//   node webui/rig_preview/measure_disocclusion.mjs <run directory>
//
// H1 found the parallax rig's limit by counting: past turnX 0.8 the reveal
// along the temple stops being scattered edge pixels and merges into one
// coherent gash. That count is the only way to compare the ellipsoid shell
// against the parallax path, so it lives in the repo now rather than in a
// scratch file.
//
// A "reveal" is a pixel that hair covered at rest and skin covers after the
// turn: the layers are occlusion-complete, so a turn never opens a hole, it
// shows correctly-painted skin where hair belongs. Everything is computed from
// alpha coverage and the manifest's z order -- the same thing the preview
// draws -- so no GPU and no colour heuristics are involved.
//
// The deformation is pulled out of index.html rather than reimplemented: a
// measurement that disagrees with the page it is measuring is worthless.
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";
import { inflateSync } from "node:zlib";

/* ---------- the page's deformation, loaded as a module ---------- */

const pageSrc = readFileSync(new URL("index.html", import.meta.url), "utf8");
const script = pageSrc.slice(pageSrc.indexOf("<script>") + 8, pageSrc.lastIndexOf("</script>"));
const stub = () => ({
  checked: false, value: "0", textContent: "", innerHTML: "",
  addEventListener() {}, append() {}, appendChild() {},
  classList: { add() {}, remove() {} }, click() {},
});
const documentStub = { getElementById: stub, createElement: stub, addEventListener() {} };
const api = new Function(
  "document", "performance", "requestAnimationFrame", "location", "fetch",
  "createImageBitmap", "URLSearchParams",
  script + ";return { buildMesh, deform, state, fitShells, SHELL_MAX_YAW, EYE_TAGS, LID_TAGS, HAIR_SHELL_TAGS };",
)(documentStub, { now: () => 0 }, () => {}, { search: "" },
  async () => { throw new Error("no fetch"); }, async () => ({}), URLSearchParams);
const { buildMesh, deform, state, fitShells, SHELL_MAX_YAW,
        EYE_TAGS, LID_TAGS, HAIR_SHELL_TAGS } = api;

/* ---------- PNG ---------- */

// Enough of the format for what save_portrait_run writes: 8-bit RGBA,
// non-interlaced. Anything else is rejected rather than silently mis-read.
function decodePNG(buf) {
  if (buf.readUInt32BE(0) !== 0x89504e47) throw new Error("not a PNG");
  let pos = 8, width = 0, height = 0;
  const idat = [];
  while (pos < buf.length) {
    const len = buf.readUInt32BE(pos);
    const type = buf.toString("latin1", pos + 4, pos + 8);
    const body = buf.subarray(pos + 8, pos + 8 + len);
    if (type === "IHDR") {
      width = body.readUInt32BE(0); height = body.readUInt32BE(4);
      if (body[8] !== 8 || body[9] !== 6 || body[12] !== 0)
        throw new Error("expected 8-bit RGBA, non-interlaced");
    } else if (type === "IDAT") idat.push(body);
    else if (type === "IEND") break;
    pos += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const bpp = 4, stride = width * bpp;
  const out = Buffer.alloc(height * stride);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const cur = out.subarray(y * stride, (y + 1) * stride);
    const prev = y ? out.subarray((y - 1) * stride, y * stride) : null;
    for (let i = 0; i < stride; i++) {
      const a = i >= bpp ? cur[i - bpp] : 0;
      const b = prev ? prev[i] : 0;
      const c = prev && i >= bpp ? prev[i - bpp] : 0;
      let v = line[i];
      if (filter === 1) v += a;
      else if (filter === 2) v += b;
      else if (filter === 3) v += (a + b) >> 1;
      else if (filter === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      } else if (filter !== 0) throw new Error("bad PNG filter " + filter);
      cur[i] = v & 0xff;
    }
  }
  const alpha = new Uint8Array(width * height);
  for (let i = 0; i < alpha.length; i++) alpha[i] = out[i * 4 + 3];
  return { width, height, alpha };
}

/* ---------- scene ---------- */

const runDir = process.argv[2];
if (!runDir) {
  console.error("usage: measure_disocclusion.mjs <run directory>");
  process.exit(2);
}
const manifestName = readdirSync(runDir).find((f) => /_rig_manifest\.json$/.test(f));
if (!manifestName) {
  console.error("no *_rig_manifest.json in " + runDir);
  process.exit(2);
}
const manifest = JSON.parse(readFileSync(join(runDir, manifestName), "utf8"));

const W = manifest.canvas.width, H = manifest.canvas.height;
state.manifest = manifest;
state.canvasW = W; state.canvasH = H;
state.breathTop = 0; state.breathBottom = H; state.chestCx = W / 2;
state.shells = fitShells(manifest.parts);
if (!state.shells) {
  console.error("no face or head layer: nothing to fit a shell to");
  process.exit(2);
}

// The non-GL half of the page's build(), which cannot be imported without a
// canvas. Kept to exactly the fields deform() reads.
const parts = manifest.parts.slice().sort((a, b) => a.z - b.z).map((spec) => ({
  spec,
  mesh: buildMesh(spec),
  isEye: EYE_TAGS.has(spec.tag),
  eyeSide: !EYE_TAGS.has(spec.tag) ? null
         : spec.tag.endsWith("l") ? "l" : spec.tag.endsWith("r") ? "r" : null,
  isLid: LID_TAGS.has(spec.tag),
  isCollar: spec.group === "body" && spec.weight.mode === "gradient_y",
  openTop: spec.xyxy[1], openBottom: spec.xyxy[3],
  shell: spec.group === "body" ? null
       : HAIR_SHELL_TAGS.has(spec.tag) ? state.shells.hair : state.shells.head,
  tex: decodePNG(readFileSync(join(runDir, spec.image))),
}));

const HAIR = new Set(["front hair", "back hair", "headwear", "hair"]);
// What may appear from under hair and count as a defect: the skull, the face,
// the pixels the remainder split gave to the head, and the neck.
const SKIN = new Set(["face", "head", "head_remainder", "ears", "nose", "neck"]);

function motionAt(turnX, shell) {
  return {
    turnX, turnY: 0, tiltRad: 0, shell,
    yaw: turnX * SHELL_MAX_YAW, pitch: 0,
    blink: { l: 0, r: 0 }, breath: 0, breathAmp: 0, chestX: 0,
    lidRatio: 0.85, lidThickness: 0.18,
    overrides: { ghost: false, neck: "gradient", collar: null },
  };
}

/** Rasterize every part's alpha in z order and keep, per pixel, the index of
 *  the topmost part covering it. Nearest sampling and a hard 0.5 alpha cut:
 *  this is a coverage question, not a rendering one. */
function ownerMap(motion) {
  const owner = new Int16Array(W * H).fill(-1);
  parts.forEach((part, pi) => {
    deform(part, 0, motion);
    const { live, uv, index } = part.mesh;
    const { width: tw, height: th, alpha } = part.tex;
    for (let t = 0; t < index.length; t += 3) {
      const i0 = index[t] * 2, i1 = index[t + 1] * 2, i2 = index[t + 2] * 2;
      const ax = live[i0], ay = live[i0 + 1];
      const bx = live[i1], by = live[i1 + 1];
      const cx = live[i2], cy = live[i2 + 1];
      const area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      if (area === 0) continue;
      const x0 = Math.max(0, Math.floor(Math.min(ax, bx, cx)));
      const x1 = Math.min(W - 1, Math.ceil(Math.max(ax, bx, cx)));
      const y0 = Math.max(0, Math.floor(Math.min(ay, by, cy)));
      const y1 = Math.min(H - 1, Math.ceil(Math.max(ay, by, cy)));
      for (let y = y0; y <= y1; y++) {
        for (let x = x0; x <= x1; x++) {
          const px = x + 0.5, py = y + 0.5;
          const b1 = ((bx - ax) * (py - ay) - (by - ay) * (px - ax)) / area;
          const b2 = ((px - ax) * (cy - ay) - (py - ay) * (cx - ax)) / area;
          if (b1 < 0 || b2 < 0 || b1 + b2 > 1) continue;
          const l0 = 1 - b1 - b2, l1 = b2, l2 = b1;
          const u = l0 * uv[i0] + l1 * uv[i1] + l2 * uv[i2];
          const v = l0 * uv[i0 + 1] + l1 * uv[i1 + 1] + l2 * uv[i2 + 1];
          const tx = Math.min(tw - 1, Math.max(0, Math.round(u * (tw - 1))));
          const ty = Math.min(th - 1, Math.max(0, Math.round(v * (th - 1))));
          if (alpha[ty * tw + tx] >= 128) owner[y * W + x] = pi;
        }
      }
    }
  });
  return owner;
}

/** Largest 4-connected region, total area, and region count of a mask. */
function regions(mask) {
  const seen = new Uint8Array(mask.length);
  const stack = new Int32Array(mask.length);
  let total = 0, largest = 0, count = 0;
  for (let i = 0; i < mask.length; i++) {
    if (!mask[i] || seen[i]) continue;
    let top = 0, size = 0;
    stack[top++] = i; seen[i] = 1;
    while (top) {
      const p = stack[--top]; size++;
      const x = p % W, y = (p - x) / W;
      const push = (q) => { if (mask[q] && !seen[q]) { seen[q] = 1; stack[top++] = q; } };
      if (x > 0) push(p - 1);
      if (x < W - 1) push(p + 1);
      if (y > 0) push(p - W);
      if (y < H - 1) push(p + W);
    }
    total += size; count++;
    if (size > largest) largest = size;
  }
  return { total, largest, count };
}

/** How far a part actually travels, averaged over its vertices. The control
 *  column: the shell is only worth anything if it reduces the reveal at the
 *  *same* apparent turn, rather than by turning the head less. */
function travelOf(tag, motion) {
  const p = parts.find((q) => q.spec.tag === tag);
  if (!p) return null;
  deform(p, 0, motion);
  const { live, rest } = p.mesh;
  let sum = 0;
  for (let v = 0; v < rest.length; v += 2) {
    const dx = live[v] - rest[v], dy = live[v + 1] - rest[v + 1];
    sum += Math.sqrt(dx * dx + dy * dy);
  }
  return sum / (rest.length / 2);
}

/** How far two parts' deformations pull the same point apart, over the box
 *  where they overlap. This is the doc's "slide" column: how far one layer
 *  travels out from behind the one in front of it. */
function slideBetween(tagA, tagB, motion) {
  const a = parts.find((p) => p.spec.tag === tagA);
  const b = parts.find((p) => p.spec.tag === tagB);
  if (!a || !b) return null;
  const box = [Math.max(a.spec.xyxy[0], b.spec.xyxy[0]), Math.max(a.spec.xyxy[1], b.spec.xyxy[1]),
               Math.min(a.spec.xyxy[2], b.spec.xyxy[2]), Math.min(a.spec.xyxy[3], b.spec.xyxy[3])];
  if (!(box[2] > box[0] && box[3] > box[1])) return null;
  // Probe meshes: the same points, each carrying one part's weight and shell.
  const probe = (src) => ({
    ...src,
    mesh: buildMesh({ ...src.spec, xyxy: box, mesh: { cell: 8 } }),
    openTop: box[1], openBottom: box[3],
  });
  const pa = probe(a), pb = probe(b);
  deform(pa, 0, motion);
  deform(pb, 0, motion);
  let max = 0;
  for (let v = 0; v < pa.mesh.live.length; v += 2) {
    const dx = (pa.mesh.live[v] - pa.mesh.rest[v]) - (pb.mesh.live[v] - pb.mesh.rest[v]);
    const dy = (pa.mesh.live[v + 1] - pa.mesh.rest[v + 1]) - (pb.mesh.live[v + 1] - pb.mesh.rest[v + 1]);
    const d = Math.sqrt(dx * dx + dy * dy);
    if (d > max) max = d;
  }
  return max;
}

/* ---------- sweep ---------- */

const turns = (process.env.TURNS || "0.2,0.4,0.6,0.8,1.0,1.5").split(",").map(Number);
const shells = (process.env.SHELLS || "0,0.5,1").split(",").map(Number);
const head = state.shells.head;

console.log("run       " + (manifest.source.run_id || manifestName));
console.log("canvas    " + W + "x" + H + ", " + parts.length + " parts");
console.log("shell     rx=" + head.rx.toFixed(1) + " ry=" + head.ry.toFixed(1) +
            " rz=" + head.rz.toFixed(1) +
            " at (" + head.cx.toFixed(0) + ", " + head.cy.toFixed(0) + ")" +
            ", max yaw " + (SHELL_MAX_YAW * 180 / Math.PI).toFixed(1) + " deg");
console.log("");
console.log("shell  turnX   nose px   slide px   revealed px   largest region   regions");

for (const shell of shells) {
  const rest = ownerMap(motionAt(0, shell));
  const restIsHair = new Uint8Array(W * H);
  for (let i = 0; i < rest.length; i++)
    restIsHair[i] = rest[i] >= 0 && HAIR.has(parts[rest[i]].spec.tag) ? 1 : 0;
  for (const turnX of turns) {
    const owner = ownerMap(motionAt(turnX, shell));
    const mask = new Uint8Array(W * H);
    for (let i = 0; i < owner.length; i++) {
      const o = owner[i];
      mask[i] = restIsHair[i] && o >= 0 && SKIN.has(parts[o].spec.tag) ? 1 : 0;
    }
    const r = regions(mask);
    const slide = slideBetween("back hair", "front hair", motionAt(turnX, shell));
    const nose = travelOf("nose", motionAt(turnX, shell));
    console.log(
      shell.toFixed(2).padStart(5) +
      turnX.toFixed(2).padStart(8) +
      (nose == null ? "       --" : nose.toFixed(1).padStart(10)) +
      (slide == null ? "        --" : slide.toFixed(1).padStart(11)) +
      String(r.total).padStart(14) +
      String(r.largest).padStart(17) +
      String(r.count).padStart(10));
  }
}
