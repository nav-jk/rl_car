let car, track, driver, pauseBtn, speedSlider, paused = false;

let agent, state;
let steps = 0, epSteps = 0, epReturn = 0, episode = 0;
let lastStats = null, lastEpReturn = null, bestEpReturn = -Infinity;

const WARMUP = 1000;          // random actions before learning starts
const MAX_EP_STEPS = 1500;    // a full lap is roughly 700-1000 steps, so 600 was too short
const TRAIN_EVERY = 2;        // one gradient step per N sim steps

function setup() {
  createCanvas(windowWidth, windowHeight);

  track = new Track();        // static, created once
  state = rst();

  pauseBtn = createButton("Pause");
  pauseBtn.position(0, 100);
  pauseBtn.mousePressed(() => paused = !paused);

  speedSlider = createSlider(1, 20, 1, 1);   // sim steps per drawn frame
  speedSlider.position(0, 130);

  // action order is [throttle, steer]
  agent = new SAC(state.length, 2, {
    actionLow: [0, -1],
    actionHigh: [1, 1],
    hidden: 128,
    batchSize: 64,
  });
}

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
  const action = steps < WARMUP ? agent.randomAction() : agent.act(state);

  // apply action and advance the physics
  driver.drive(action[0], action[1]);
  car.envForces();
  car.update();
  driver.update();            // sensors follow the new car position

  // observe (each of these is called exactly once per step)
  const next = driver.state;
  const crashed = driver.isOffTrack();
  const lap = driver.hasCompletedLap();
  const r = driver.reward(crashed, lap);

  epSteps++; steps++; epReturn += r;

  // only a crash is a true terminal; a finished lap or the step limit is not
  agent.remember(state, action, r, next, crashed);

  if (steps >= WARMUP && steps % TRAIN_EVERY === 0) {
    const st = agent.train();
    if (st) lastStats = st;
  }

  if (crashed || lap || epSteps >= MAX_EP_STEPS) {
    episode++;
    lastEpReturn = epReturn;
    bestEpReturn = Math.max(bestEpReturn, epReturn);
    console.log(`ep ${episode}  return ${epReturn.toFixed(1)}  len ${epSteps}  ` +
      `${crashed ? "crash" : lap ? "LAP" : "timeout"}  steps ${steps}`);
    state = rst();
    epSteps = 0; epReturn = 0;
  } else {
    state = next;
  }
}

function draw() {
  if (!paused) {
    for (let i = 0; i < speedSlider.value(); i++) rlStep();
  }

  background(220);
  track.show();
  driver.show();
  car.show();

  const lines = [
    `step ${steps}  episode ${episode}  ep step ${epSteps}`,
    `ep return ${epReturn.toFixed(1)}  last ${lastEpReturn === null ? "-" : lastEpReturn.toFixed(1)}  best ${bestEpReturn === -Infinity ? "-" : bestEpReturn.toFixed(1)}`,
    steps < WARMUP ? `warm-up ${steps}/${WARMUP}` :
      (lastStats ? `alpha ${lastStats.alpha.toFixed(3)}  critic ${lastStats.criticLoss.toFixed(3)}  actor ${lastStats.actorLoss.toFixed(3)}` : ""),
    `state ${state.map(v => v.toFixed(2)).join(" ")}`,
  ];
  push();
  noStroke();
  fill(0);
  lines.forEach((t, i) => text(t, width - 420, 100 + i * 16));
  pop();
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