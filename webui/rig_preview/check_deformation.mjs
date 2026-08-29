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
      id, checked: id === "doBlink" || id === "doBreathe",   // mirrors the page defaults
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
  src + "\nreturn { weightAt, smoothstep, buildMesh, deform, state, startBlink, blinkAmount, EYE_TAGS, breathRamp };",
)(document, performance, requestAnimationFrame, location, fetch, createImageBitmap, URLSearchParams);

const { weightAt, buildMesh, deform, state, startBlink, blinkAmount, breathRamp } = api;

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
const NO_OVERRIDE = { ghost: false, neck: "gradient", collar: null };
const still = { turnX: 0, turnY: 0, tiltRad: 0, blink: { l: 0, r: 0 }, breath: 0,
                breathAmp: 8, chestX: 0.004, lidRatio: 0.85, lidThickness: 0.18,
                overrides: NO_OVERRIDE };
// Chest band: torso from y=450 down to the planted pivot at y=950.
state.breathTop = 450; state.breathBottom = 950; state.chestCx = 500;

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
                 { isEye: true, eyeSide: "l", eyeCenterY: 300,
                   openTop: 280, openBottom: 320 });
const lash = part({ tag: "eyelashl", group: "head", depth: 0.34, xyxy: [438, 284, 482, 300],
                    mesh: { cell: 8 }, weight: { mode: "constant", value: 1 } },
                  { isEye: true, isLid: true, eyeSide: "l", eyeCenterY: 300,
                    openTop: 280, openBottom: 320 });

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
// opening 280..320 with ratio 0.85 puts the lid at 314, so the top of the eye
// travels the full 34px down onto it rather than 20px to the centre.
check("the left eye collapses onto the lid line", near(closed.dy, 34, 1e-6),
      `dy=${closed.dy}`);
deform(eye, 0, { ...still, blink: { l: 0, r: 1 } });
check("a right wink leaves the left eye open", near(shiftAt(eye, 280).dy, 0));

check("a closed eye still has a lid: the lash does not collapse to nothing",
      (() => {
        deform(lash, 0, { ...still, blink: { l: 1, r: 0 } });
        const ys = [];
        for (let v = 1; v < lash.mesh.live.length; v += 2) ys.push(lash.mesh.live[v]);
        return Math.max(...ys) - Math.min(...ys) > 1.5;
      })(),
      "the lash scaled to zero height and nothing would be drawn");
check("the white and the iris do vanish behind it",
      (() => {
        deform(eye, 0, { ...still, blink: { l: 1, r: 0 } });
        const ys = [];
        for (let v = 1; v < eye.mesh.live.length; v += 2) ys.push(eye.mesh.live[v]);
        return Math.max(...ys) - Math.min(...ys) < 0.001;
      })());
check("the eye closes onto the lower lid, not the middle of the opening",
      (() => {
        const mean = (arr) => { let s = 0, n = 0;
          for (let v = 1; v < arr.length; v += 2) { s += arr[v]; n++; } return s / n; };
        deform(lash, 0, { ...still, blink: { l: 1, r: 0 } });
        const closed = mean(lash.mesh.live);
        // opening 280..320, ratio 0.85 -> lid at 314, well below the centre 300
        return closed > 306 && closed < 320;
      })(),
      "the lid landed near the centre, which reads as a squint");
check("the lid line follows its ratio",
      (() => {
        const mean = (arr) => { let s = 0, n = 0;
          for (let v = 1; v < arr.length; v += 2) { s += arr[v]; n++; } return s / n; };
        deform(lash, 0, { ...still, blink: { l: 1, r: 0 }, lidRatio: 0.5 });
        const mid = mean(lash.mesh.live);
        deform(lash, 0, { ...still, blink: { l: 1, r: 0 }, lidRatio: 1.0 });
        return mean(lash.mesh.live) > mid + 8;
      })());
check("the closed lid travels almost all the way to the lid line",
      (() => {
        const mean = (arr) => { let s = 0, n = 0;
          for (let v = 1; v < arr.length; v += 2) { s += arr[v]; n++; } return s / n; };
        const lid = 280 + 0.85 * (320 - 280);
        const open = mean(lash.mesh.rest);
        deform(lash, 0, { ...still, blink: { l: 1, r: 0 } });
        const closed = mean(lash.mesh.live);
        return Math.abs(closed - lid) < 0.25 * Math.abs(open - lid);
      })());
check("a thicker lid keeps more of the lash",
      (() => {
        const height = (t) => {
          deform(lash, 0, { ...still, blink: { l: 1, r: 0 }, lidThickness: t });
          const ys = [];
          for (let v = 1; v < lash.mesh.live.length; v += 2) ys.push(lash.mesh.live[v]);
          return Math.max(...ys) - Math.min(...ys);
        };
        return height(0.5) > height(0.18) && height(0.18) > height(0.05);
      })());
check("a half-blink is between open and closed",
      (() => {
        deform(lash, 0, { ...still, blink: { l: 0.5, r: 0 } });
        const ys = [];
        for (let v = 1; v < lash.mesh.live.length; v += 2) ys.push(lash.mesh.live[v]);
        const h = Math.max(...ys) - Math.min(...ys);
        return h > 6 && h < 16;
      })());

console.log("\nblink envelope");
state.blinkPhase = null;
startBlink(0, ["l", "r"]);
const at = (ms) => blinkAmount(ms).l;
check("starts open", near(at(0), 0));
check("fully closed after the close phase", near(at(80), 1));
check("still closed through the hold", near(at(400), 1));
check("open again after the whole envelope", at(581) === 0);

console.log("\nbreathing (one continuous field)");
const breathe = { ...still, breath: 1 };
const torso = part({ tag: "topwear", group: "body", depth: 0.74, xyxy: [300, 450, 700, 950],
                     mesh: { cell: 50 }, weight: { mode: "constant", value: 0.16 } });
deform(faceNear, 0, breathe);
deform(neck, 0, breathe);
deform(torso, 0, breathe);
const headLift = -shiftAt(faceNear, 300).dy;
const neckTopLift = -shiftAt(neck, 600).dy;
const neckBottomLift = -shiftAt(neck, 700).dy;
const chestTopLift = -shiftAt(torso, 450).dy;
const chestBottomLift = -shiftAt(torso, 950).dy;
check("the head rises", near(headLift, 8, 1e-6), `lift=${headLift}`);
check("the chest top rises by the same amount as the head",
      near(chestTopLift, headLift, 1e-6), `chest=${chestTopLift} head=${headLift}`);
check("the bottom of the torso stays planted", near(chestBottomLift, 0, 1e-9),
      `lift=${chestBottomLift}`);
// A neck sitting above the chest line, as in the real runs (neck rows 261-463
// against a topwear top of 363) rather than buried inside the torso.
const neckHigh = part({ tag: "neck", group: "neck", depth: 0.7, xyxy: [470, 300, 530, 430],
                        mesh: { cell: 25 },
                        weight: { mode: "gradient_y", top: 0.55, bottom: 0, y_top: 300, y_bottom: 430 } });
deform(neckHigh, 0, breathe);
const hiTop = -shiftAt(neckHigh, 300).dy, hiBottom = -shiftAt(neckHigh, 430).dy;
check("the neck does not stretch: it rides whole with the head",
      near(hiTop, headLift, 1e-6) && near(hiBottom, headLift, 1e-6),
      `top=${hiTop} bottom=${hiBottom} head=${headLift}`);
check("a neck buried in the chest stays continuous with the torso around it",
      near(neckBottomLift, -shiftAt(torso, 700).dy, 1e-6),
      `neck=${neckBottomLift} torso=${-shiftAt(torso, 700).dy}`);
check("that buried part still lifts less than the head, as a chest point should",
      neckTopLift < headLift && neckTopLift > 0, `neck=${neckTopLift} head=${headLift}`);
check("the field is monotone through the chest",
      -shiftAt(torso, 550).dy > -shiftAt(torso, 750).dy);
check("ramp is 1 above the chest and 0 at the planted bottom",
      breathRamp(100) === 1 && breathRamp(950) === 0);
check("the ribcage widens as it rises",
      Math.abs(shiftAt(torso, 460).dx) > 0, `dx=${shiftAt(torso, 460).dx}`);
deform(faceNear, 0, breathe);
check("the head does not widen", near(shiftAt(faceNear, 300).dx, 0, 1e-9));

console.log("\ncollar ramp");
const collar = part({ tag: "topwear", group: "body", depth: 0.74, xyxy: [300, 430, 700, 950],
                      mesh: { cell: 40 },
                      weight: { mode: "gradient_y", top: 0.45, bottom: 0.16,
                                y_top: 430, y_bottom: 560 } },
                    { isCollar: true });
deform(collar, 0, turn);
const collarTop = shiftAt(collar, 430).dx, hem = shiftAt(collar, 950).dx;
check("the collar follows the head more than the hem does", collarTop > hem * 1.5,
      `collar=${collarTop.toFixed(2)} hem=${hem.toFixed(2)}`);
check("the hem still moves with the body", hem > 0, `hem=${hem}`);
deform(collar, 0, { ...turn, overrides: { ...NO_OVERRIDE, collar: 1.0 } });
const lively = shiftAt(collar, 430).dx;
check("the collar slider re-aims the top of the ramp", lively > collarTop * 1.8,
      `at1.0=${lively.toFixed(2)} baked=${collarTop.toFixed(2)}`);
check("the slider leaves the hem alone",
      near(shiftAt(collar, 950).dx, hem, 1e-9));
deform(collar, 0, { ...turn, overrides: { ...NO_OVERRIDE, collar: 0.16 } });
check("flattening the slider makes the garment rigid again",
      near(shiftAt(collar, 430).dx, shiftAt(collar, 950).dx, 1e-6));

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
