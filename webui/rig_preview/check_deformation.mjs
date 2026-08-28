// Headless check of the rig preview's deformation math.
//
//   node webui/rig_preview/check_deformation.mjs webui/rig_preview/index.html
//
// Pulls the <script> out of index.html, runs it against DOM stubs, and
// exercises the pure functions. The preview is a single self-contained file
// with no build step, so this is the only thing standing between a typo in the
// weight math and a wrong answer to H1-H4 -- a rig that looks plausible while
// deforming incorrectly is worse than one that visibly breaks.
import { readFileSync } from "node:fs";

const html = readFileSync(process.argv[2] || new URL("index.html", import.meta.url), "utf8");
const src = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));

const controls = {};
function control(id) {
  if (!controls[id]) {
    controls[id] = {
      id, checked: id === "autoIdle" || id === "doBlink" || id === "doBreathe",
      value: id === "neckMode" ? "gradient" : "0",
      textContent: "", innerHTML: "",
      addEventListener() {}, append() {}, appendChild() {},
      classList: { add() {}, remove() {} },
      click() {},
    };
  }
  return controls[id];
}
const document = {
  getElementById: control,
  createElement: () => control("_tmp" + Math.random()),
  addEventListener() {},
};
const performance = { now: () => 0 };
const requestAnimationFrame = () => {};
const location = { search: "" };
const fetch = async () => { throw new Error("no fetch in harness"); };
const createImageBitmap = async () => ({});

const api = new Function(
  "document", "performance", "requestAnimationFrame", "location", "fetch",
  "createImageBitmap", "URLSearchParams",
  src + "\nreturn { weightAt, smoothstep, buildMesh, deform, state, startBlink, blinkAmount, EYE_TAGS };",
)(document, performance, requestAnimationFrame, location, fetch, createImageBitmap, URLSearchParams);

const { weightAt, buildMesh, deform, state, startBlink, blinkAmount } = api;

let failures = 0;
function check(name, cond, detail = "") {
  if (cond) console.log(`  ok   ${name}`);
  else { console.log(`  FAIL ${name} ${detail}`); failures++; }
}
const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

console.log("weightAt");
const grad = { mode: "gradient_y", top: 0.55, bottom: 0.0, y_top: 100, y_bottom: 200 };
check("top of the band follows the head at full gradient weight", near(weightAt(grad, 100), 0.55));
check("bottom of the band is detached", near(weightAt(grad, 200), 0.0));
check("midpoint is halfway", near(weightAt(grad, 150), 0.275));
check("above the band clamps to top", near(weightAt(grad, 0), 0.55));
check("below the band clamps to bottom", near(weightAt(grad, 999), 0.0));
check("constant mode is flat", near(weightAt({ mode: "constant", value: 0.16 }, 12), 0.16));

// A minimal scene: one neck part with the manifest gradient, two head parts at
// different depths, one eye part.
state.canvasW = 1000; state.canvasH = 1000;
state.manifest = {
  canvas: { width: 1000, height: 1000 },
  anchors: { neck_pivot: [500, 700], body_pivot: [500, 950], eye_left: [460, 300] },
  motion: { breathing: { period_s: 4, amplitude_px: 3 }, head_tilt: { max_deg: 2 },
            blink: { close_s: 0.08, hold_s: 0.34, open_s: 0.16, interval_s: [1.6, 5.4] } },
};

function part(spec, extra = {}) {
  const mesh = buildMesh(spec);
  return { spec, mesh, isEye: false, eyeSide: null,
           eyeCenterY: (spec.xyxy[1] + spec.xyxy[3]) / 2, ...extra };
}
const NO_OVERRIDE = { ghost: false, neck: "gradient" };
const still = { turnX: 0, turnY: 0, tiltRad: 0, blink: { l: 0, r: 0 }, breath: 0,
                overrides: NO_OVERRIDE };

const neck = part({ tag: "neck", group: "neck", depth: 0.7, xyxy: [470, 600, 530, 700],
                    mesh: { cell: 25 },
                    weight: { mode: "gradient_y", top: 0.55, bottom: 0, y_top: 600, y_bottom: 700 } });
const faceNear = part({ tag: "face", group: "head", depth: 0.2, xyxy: [400, 200, 600, 400],
                        mesh: { cell: 50 }, weight: { mode: "constant", value: 1 } });
const hairFar = part({ tag: "back hair", group: "head", depth: 0.95, xyxy: [400, 200, 600, 400],
                       mesh: { cell: 50 }, weight: { mode: "constant", value: 1 } });
const headRem = part({ tag: "head_remainder", group: "head", depth: 0.63,
                       xyxy: [380, 210, 420, 260], mesh: { cell: 20 },
                       weight: { mode: "constant", value: 1 } });
const eye = part({ tag: "eyewhitel", group: "head", depth: 0.35, xyxy: [440, 280, 480, 320],
                   mesh: { cell: 10 }, weight: { mode: "constant", value: 1 } },
                 { isEye: true, eyeSide: "l", eyeCenterY: 300 });

// dx of the vertex nearest a given y, versus rest position.
function shiftAt(p, wantY) {
  let best = 0, bestD = Infinity;
  for (let i = 0, v = 0; v < p.mesh.rest.length; i++, v += 2) {
    const d = Math.abs(p.mesh.rest[v + 1] - wantY);
    if (d < bestD) { bestD = d; best = v; }
  }
  return { dx: p.mesh.live[best] - p.mesh.rest[best],
           dy: p.mesh.live[best + 1] - p.mesh.rest[best + 1] };
}

console.log("\nneck gradient (H3)");
const turn = { ...still, turnX: 1 };
deform(neck, 0, turn);
const neckTop = shiftAt(neck, 600), neckBottom = shiftAt(neck, 700);
check("top of the neck follows the head", neckTop.dx > 1, `dx=${neckTop.dx}`);
check("bottom of the neck stays put", near(neckBottom.dx, 0, 1e-9), `dx=${neckBottom.dx}`);
check("the neck deforms continuously, not rigidly",
      neckTop.dx > shiftAt(neck, 650).dx && shiftAt(neck, 650).dx > neckBottom.dx);

deform(neck, 0, { ...turn, overrides: { ghost: false, neck: "rigid" } });
check("rigid override moves the whole neck together",
      near(shiftAt(neck, 600).dx, shiftAt(neck, 700).dx));
deform(neck, 0, { ...turn, overrides: { ghost: false, neck: "detached" } });
check("detached override pins the whole neck", near(shiftAt(neck, 650).dx, 0));

console.log("\ndepth parallax (H1/H5)");
deform(faceNear, 0, turn);
deform(hairFar, 0, turn);
const nearShift = shiftAt(faceNear, 300).dx, farShift = shiftAt(hairFar, 300).dx;
check("near parts travel further than far ones", nearShift > farShift,
      `near=${nearShift.toFixed(2)} far=${farShift.toFixed(2)}`);
check("far parts still move a little", farShift > 0, `far=${farShift}`);
check("parallax is a fraction of the canvas, not a fixed pixel count",
      nearShift > 20 && nearShift < 100, `near=${nearShift.toFixed(2)}`);

console.log("\nghost silhouette toggle (H2)");
deform(headRem, 0, turn);
const withHead = shiftAt(headRem, 230).dx;
deform(headRem, 0, { ...turn, overrides: { ghost: true, neck: "gradient" } });
const withBody = shiftAt(headRem, 230).dx;
check("head_remainder follows the head by default", withHead > 20, `dx=${withHead.toFixed(2)}`);
check("the ghost toggle strands it with the body", withBody < withHead * 0.25,
      `head=${withHead.toFixed(2)} body=${withBody.toFixed(2)}`);

console.log("\nblink (H4)");
deform(eye, 0, { ...still, blink: { l: 1, r: 0 } });
const closed = shiftAt(eye, 280);
check("the left eye collapses onto its centre line", near(closed.dy, 20, 1e-6),
      `dy=${closed.dy}`);
deform(eye, 0, { ...still, blink: { l: 0, r: 1 } });
check("a right wink leaves the left eye open", near(shiftAt(eye, 280).dy, 0));

console.log("\nblink envelope");
state.blinkPhase = null;
startBlink(0, ["l", "r"]);
const at = (ms) => blinkAmount(ms).l;
check("starts open", near(at(0), 0));
check("fully closed after the close phase", near(at(80), 1));
check("still closed through the hold", near(at(400), 1));
check("open again after the whole envelope", at(581) === 0);

console.log("\ntilt about the neck pivot");
state.blinkPhase = null;
deform(faceNear, 0, { ...still, tiltRad: 10 * Math.PI / 180 });
const tilted = shiftAt(faceNear, 300);
check("a head part swings sideways when tilted", Math.abs(tilted.dx) > 10,
      `dx=${tilted.dx.toFixed(2)}`);
deform(neck, 0, { ...still, tiltRad: 10 * Math.PI / 180 });
check("the bottom of the neck does not swing", near(shiftAt(neck, 700).dx, 0, 1e-9));

console.log(failures ? `\n${failures} FAILED` : "\nall checks passed");
process.exit(failures ? 1 : 0);
