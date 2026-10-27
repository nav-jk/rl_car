let car, track, driver, paused = false, busy = false;   // busy = save/load in progress

let agent, state;
let steps = 0;            // learning steps (what warm-up and the save file count)
let simTick = 0;          // every simulated frame, including greedy ones (trail age)
let epSteps = 0, epReturn = 0, episode = 0, totalLaps = 0;
let lastStats = null, lastEpReturn = null, bestEpReturn = -Infinity;
let history = { epReturn: [], epLen: [], outcome: [] };

const WARMUP = 1000;          // random actions before learning starts
const MAX_EP_STEPS = 1500;    // a full lap is roughly 700-1000 steps, so 600 was too short
const TRAIN_EVERY = 2;        // one gradient step per N sim steps

// ---- UI state ----
const PANEL_X = 500;          // the track occupies x < 500, the dashboard lives to its right
let trail, charts, speedSlider, pauseBtn, trailChk, bufferChk, greedyChk, fileInput;
let statusMsg = '', statusUntil = 0;

function setup() {
  createCanvas(windowWidth, windowHeight);

  track = new Track();        // static, created once
  state = rst();

  // action order is [throttle, steer]
  agent = new SAC(state.length, 2, {
    actionLow: [0, -1],
    actionHigh: [1, 1],
    hidden: 128,
    batchSize: 64,
  });

  trail = new Trail();
  charts = {
    throttle: new LiveChart({ title: 'throttle', len: 300, min: 0, max: 1, series: [{ color: '#d97706' }] }),
    steer:    new LiveChart({ title: 'steer', len: 300, min: -1, max: 1, series: [{ color: '#2563eb' }] }),
    reward:   new LiveChart({ title: 'reward / step', len: 300, series: [{ color: '#059669' }], digits: 3, robust: true }),
    epReturn: new LiveChart({ title: 'episode return (+ avg 10)', len: 200,
                              series: [{ color: '#9ca3af' }, { color: '#dc2626' }], digits: 1 }),
    critic:   new LiveChart({ title: 'critic loss', len: 300, series: [{ color: '#7c3aed' }], digits: 3, robust: true }),
    alpha:    new LiveChart({ title: 'alpha (temperature)', len: 300, series: [{ color: '#0891b2' }], digits: 3 }),
  };
  buildUI();
}

function windowResized() { resizeCanvas(windowWidth, windowHeight); }

function buildUI() {
  const box = createDiv();
  box.id('controls');

  pauseBtn = createButton('Pause');
  pauseBtn.parent(box);
  pauseBtn.mousePressed(() => { paused = !paused; pauseBtn.html(paused ? 'Resume' : 'Pause'); });

  const saveBtn = createButton('Save');
  saveBtn.parent(box);
  saveBtn.mousePressed(saveRun);

  const loadBtn = createButton('Load');
  loadBtn.parent(box);
  loadBtn.mousePressed(() => fileInput.elt.click());

  fileInput = createElement('input');
  fileInput.attribute('type', 'file');
  fileInput.attribute('accept', '.json,application/json');
  fileInput.hide();
  fileInput.parent(box);
  fileInput.elt.addEventListener('change', async (e) => {
    const f = e.target.files[0];
    e.target.value = '';                     // allow re-loading the same file
    if (f) await loadRunFile(f);
  });

  bufferChk = createCheckbox('save replay buffer', true);
  bufferChk.parent(box);
  trailChk = createCheckbox('trail', true);
  trailChk.parent(box);
  greedyChk = createCheckbox('greedy (no learning)', false);
  greedyChk.parent(box);

  const lbl = createSpan('speed');
  lbl.parent(box);
  speedSlider = createSlider(1, 20, 1, 1);   // sim steps per drawn frame
  speedSlider.parent(box);
}

function setStatus(msg, ms = 6000) { statusMsg = msg; statusUntil = millis() + ms; }

// Start a new episode and return the first observation
function rst() {
  car = new Car(30, 200);
  driver = new Driver(car, track);
  driver.sensorInit();        // only after the track is set
  driver.resetProgress();
  driver.update();            // one sense pass so the first state is real
  return driver.state;
}

// One RL step = choose action -> advance sim one frame -> observe -> learn
function rlStep() {
  const greedy = greedyChk.checked();
  const action = greedy ? agent.act(state, true)
               : steps < WARMUP ? agent.randomAction() : agent.act(state);

  // apply action and advance the physics
  const px = car.pos.x, py = car.pos.y;
  driver.drive(action[0], action[1]);
  car.envForces();
  car.update();
  driver.update();            // sensors follow the new car position
  simTick++;
  trail.add(px, py, car.pos.x, car.pos.y, simTick);

  // observe (each of these is called exactly once per step)
  const next = driver.state;
  const crashed = driver.isOffTrack();
  const lap = driver.hasCompletedLap();
  const r = driver.reward(crashed, lap);

  epSteps++; epReturn += r;
  charts.throttle.push(action[0]);
  charts.steer.push(action[1]);
  charts.reward.push(r);
  if (lap) totalLaps++;

  if (!greedy) {
    steps++;
    // only a crash is a true terminal; a finished lap or the step limit is not
    agent.remember(state, action, r, next, crashed);

    if (steps >= WARMUP && steps % TRAIN_EVERY === 0) {
      const st = agent.train();
      if (st) {
        lastStats = st;
        charts.critic.push(st.criticLoss);
        charts.alpha.push(st.alpha);
      }
    }
  }

  if (crashed || lap || epSteps >= MAX_EP_STEPS) {
    const outcome = crashed ? 'crash' : lap ? 'LAP' : 'timeout';
    episode++;
    lastEpReturn = epReturn;
    bestEpReturn = Math.max(bestEpReturn, epReturn);
    history.epReturn.push(+epReturn.toFixed(3));
    history.epLen.push(epSteps);
    history.outcome.push(outcome);
    charts.epReturn.push(epReturn, 0);
    const recent = history.epReturn.slice(-10);
    charts.epReturn.push(recent.reduce((a, b) => a + b, 0) / recent.length, 1);
    console.log(`ep ${episode}  return ${epReturn.toFixed(1)}  len ${epSteps}  ${outcome}  steps ${steps}`);
    state = rst();
    epSteps = 0; epReturn = 0;
  } else {
    state = next;
  }
}

function draw() {
  if (!paused && !busy) {
    for (let i = 0; i < speedSlider.value(); i++) rlStep();
  }

  background(220);
  track.show();
  if (trailChk.checked()) trail.draw(simTick);
  driver.show();
  car.show();

  drawDashboard();
}

function drawDashboard() {
  const lines = [
    `step ${steps}  episode ${episode}  ep step ${epSteps}  laps ${totalLaps}`,
    `ep return ${epReturn.toFixed(1)}  last ${lastEpReturn === null ? '-' : lastEpReturn.toFixed(1)}  best ${bestEpReturn === -Infinity ? '-' : bestEpReturn.toFixed(1)}`,
    greedyChk.checked() ? 'greedy mode: watching the policy, nothing is learned' :
      steps < WARMUP ? `warm-up ${steps}/${WARMUP}` :
      (lastStats ? `alpha ${lastStats.alpha.toFixed(3)}  critic ${lastStats.criticLoss.toFixed(3)}  actor ${lastStats.actorLoss.toFixed(3)}` : ''),
    millis() < statusUntil ? statusMsg : '',
  ];
  push();
  noStroke();
  textSize(12);
  textAlign(LEFT, TOP);
  lines.forEach((t, i) => { fill(i === 3 ? '#b45309' : 0); text(t, PANEL_X, 74 + i * 16); });
  pop();

  // chart grid: 2 columns x 3 rows
  const x0 = PANEL_X, y0 = 150, gap = 10;
  const cw = Math.max(180, Math.min(380, (width - x0 - 16 - gap) / 2));
  const ch = clamp((height - y0 - 10 - 2 * gap) / 3, 60, 120);
  const grid = [['throttle', 'steer'], ['reward', 'epReturn'], ['critic', 'alpha']];
  grid.forEach((row, r) => row.forEach((k, c) => charts[k].draw(x0 + c * (cw + gap), y0 + r * (ch + gap), cw, ch)));
}

// ---------------------------------------------------------------------------
// Save / load
// ---------------------------------------------------------------------------
// One JSON file holds the agent (networks, alpha, Adam state, optionally the replay buffer)
// plus everything the sketch tracked during training (counters, history, chart data).
async function saveRun() {
  if (busy) return;
  busy = true;
  try {
    setStatus('saving...');
    const run = {
      format: 'rl-car-run', version: 1, savedAt: new Date().toISOString(),
      progress: {
        steps, episode, totalLaps, simTick,
        bestEpReturn: Number.isFinite(bestEpReturn) ? bestEpReturn : null,
        lastEpReturn, lastStats,
      },
      constants: { WARMUP, MAX_EP_STEPS, TRAIN_EVERY },
      history,
      charts: Object.fromEntries(Object.entries(charts).map(([k, c]) => [k, c.toJSON()])),
      agent: await agent.exportState({ includeBuffer: bufferChk.checked() }),
    };
    const blob = new Blob([JSON.stringify(run)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `rl_car_step${steps}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 10000);
    setStatus(`saved ${a.download} (${(blob.size / 1e6).toFixed(1)} MB)`);
  } catch (e) {
    console.error(e);
    setStatus('save failed: ' + e.message, 15000);
  } finally {
    busy = false;
  }
}

async function loadRunFile(file) {
  if (busy) return;
  busy = true;
  try {
    setStatus('loading...', 60000);
    const run = JSON.parse(await file.text());
    if (run.format !== 'rl-car-run') throw new Error('not an rl_car save file');

    const warnings = await agent.importState(run.agent);

    const p = run.progress;
    steps = p.steps; episode = p.episode; totalLaps = p.totalLaps ?? 0; simTick = p.simTick ?? 0;
    bestEpReturn = p.bestEpReturn ?? -Infinity;
    lastEpReturn = p.lastEpReturn ?? null;
    lastStats = p.lastStats ?? null;
    history = run.history ?? { epReturn: [], epLen: [], outcome: [] };
    for (const [k, c] of Object.entries(charts)) c.load(run.charts && run.charts[k]);

    trail.clear();
    state = rst();            // the old episode is meaningless for the loaded policy
    epSteps = 0; epReturn = 0;

    setStatus(`loaded ${file.name}: step ${steps}, episode ${episode}` +
      (warnings.length ? `  (${warnings.join('; ')})` : ''), 15000);
    warnings.forEach(w => console.warn(w));
  } catch (e) {
    console.error(e);
    setStatus('load failed: ' + e.message, 15000);
  } finally {
    busy = false;
  }
}

class Track {
  constructor() {
    this.outer_pts = [[10, 10], [470, 10], [470, 470], [10, 470]];
    this.inner_pts = [[70, 70], [70, 400], [400, 400], [400, 70]];
    this.out_vertices = this.outer_pts.length;
    this.in_vertices = this.inner_pts.length;

    this.obstacles = [];
    this.checkpoints = [];

    for (let i = 0; i < this.in_vertices; i++) {
      this.addStaticObstacle(this.inner_pts[i][0], this.inner_pts[i][1], this.inner_pts[(i + 1) % this.in_vertices][0], this.inner_pts[(i + 1) % this.in_vertices][1]);
    }
    for (let i = 0; i < this.out_vertices; i++) {
      this.addStaticObstacle(this.outer_pts[i][0], this.outer_pts[i][1], this.outer_pts[(i + 1) % this.out_vertices][0], this.outer_pts[(i + 1) % this.out_vertices][1]);
    }
  }
  addStaticObstacle(x1, y1, x2, y2) {
    this.obstacles.push({ start: { x: x1, y: y1 }, end: { x: x2, y: y2 } });
  }

  show() {
    beginShape();
    for (let i of this.outer_pts) vertex(i[0], i[1]);
    beginContour();
    for (let i of this.inner_pts) vertex(i[0], i[1]);
    endContour(CLOSE);
    endShape(CLOSE);
  }

  checkLeftTrackLimits(pts) {
    for (let p of pts) {
      if (isInside(this.inner_pts, p.x, p.y)) return true;
      if (!isInside(this.outer_pts, p.x, p.y)) return true;
    }
    return false;
  }
}