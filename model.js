// Soft Actor-Critic (continuous actions) in TensorFlow.js
// Browser: <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>  (load BEFORE this file)

const tf = typeof window !== 'undefined' ? window.tf : require('@tensorflow/tfjs');

// ---------- Replay buffer ----------
class ReplayBuffer {
  constructor(capacity = 100000) {
    this.capacity = capacity;
    this.data = [];
    this.idx = 0;
  }
  add(s, a, r, s2, done) {
    const item = { s, a, r, s2, d: done ? 1 : 0 };
    if (this.data.length < this.capacity) this.data.push(item);
    else this.data[this.idx] = item;
    this.idx = (this.idx + 1) % this.capacity;
  }
  get size() { return this.data.length; }
  sample(n) {
    const b = [];
    for (let i = 0; i < n; i++) b.push(this.data[Math.floor(Math.random() * this.data.length)]);
    return {
      s: tf.tensor2d(b.map(x => x.s)),
      a: tf.tensor2d(b.map(x => x.a)),
      r: tf.tensor2d(b.map(x => [x.r])),
      s2: tf.tensor2d(b.map(x => x.s2)),
      d: tf.tensor2d(b.map(x => [x.d])),
    };
  }
}

// ---------- Networks ----------
function buildActor(stateDim, actionDim, hidden = 256) {
  const input = tf.input({ shape: [stateDim] });
  let x = tf.layers.dense({ units: hidden, activation: 'relu' }).apply(input);
  x = tf.layers.dense({ units: hidden, activation: 'relu' }).apply(x);
  const mean = tf.layers.dense({ units: actionDim }).apply(x);
  const logStd = tf.layers.dense({ units: actionDim }).apply(x);
  return tf.model({ inputs: input, outputs: [mean, logStd] });
}

function buildCritic(stateDim, actionDim, hidden = 256) {
  const input = tf.input({ shape: [stateDim + actionDim] });
  let x = tf.layers.dense({ units: hidden, activation: 'relu' }).apply(input);
  x = tf.layers.dense({ units: hidden, activation: 'relu' }).apply(x);
  const q = tf.layers.dense({ units: 1 }).apply(x);
  return tf.model({ inputs: input, outputs: q });
}

const LOG_2PI = Math.log(2 * Math.PI);

// ---------- SAC agent ----------
// Options:
//   actionLow / actionHigh : arrays, one entry per action dimension (e.g. [0, -1] / [1, 1])
//   actionScale            : fallback if low/high not given -> range [-scale, scale] for every dim
//   hidden, batchSize, lr, gamma, tau, bufferSize, targetEntropy
class SAC {
  constructor(stateDim, actionDim, opts = {}) {
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.gamma = opts.gamma ?? 0.99;
    this.tau = opts.tau ?? 0.005;
    this.batchSize = opts.batchSize ?? 256;
    this.targetEntropy = opts.targetEntropy ?? -actionDim;

    // per-dimension action range
    const scale = opts.actionScale ?? 1.0;
    this.actionLow = opts.actionLow ?? new Array(actionDim).fill(-scale);
    this.actionHigh = opts.actionHigh ?? new Array(actionDim).fill(scale);
    if (this.actionLow.length !== actionDim || this.actionHigh.length !== actionDim) {
      throw new Error('actionLow/actionHigh must have length ' + actionDim);
    }
    this._mid = this.actionLow.map((l, i) => (l + this.actionHigh[i]) / 2);
    this._half = this.actionLow.map((l, i) => (this.actionHigh[i] - l) / 2);

    const hidden = opts.hidden ?? 256;
    this.actor = buildActor(stateDim, actionDim, hidden);
    this.q1 = buildCritic(stateDim, actionDim, hidden);
    this.q2 = buildCritic(stateDim, actionDim, hidden);
    this.q1T = buildCritic(stateDim, actionDim, hidden);
    this.q2T = buildCritic(stateDim, actionDim, hidden);
    this.q1T.setWeights(this.q1.getWeights());
    this.q2T.setWeights(this.q2.getWeights());

    this.logAlpha = tf.variable(tf.scalar(0.0));

    const lr = opts.lr ?? 3e-4;
    this.actorOpt = tf.train.adam(lr);
    this.criticOpt = tf.train.adam(lr);
    this.alphaOpt = tf.train.adam(lr);

    this.buffer = new ReplayBuffer(opts.bufferSize ?? 100000);
  }

  // network space [-1, 1] <-> environment space [low, high]
  _toEnv(a) { return a.map((v, i) => this._mid[i] + v * this._half[i]); }
  _toNorm(a) {
    return a.map((v, i) => {
      const n = this._half[i] === 0 ? 0 : (v - this._mid[i]) / this._half[i];
      return Math.max(-1, Math.min(1, n));
    });
  }

  // uniform random action in the environment range (for warm-up)
  randomAction() {
    return this._toEnv(Array.from({ length: this.actionDim }, () => Math.random() * 2 - 1));
  }

  // tanh-squashed Gaussian sample + log-prob
  _sample(state) {
    const [mean, logStdRaw] = this.actor.apply(state);
    const logStd = tf.clipByValue(logStdRaw, -20, 2);
    const std = tf.exp(logStd);
    const u = mean.add(std.mul(tf.randomNormal(mean.shape)));
    const a = tf.tanh(u);
    const logProbGauss = u.sub(mean).div(std).square().mul(-0.5).sub(logStd).sub(0.5 * LOG_2PI);
    const correction = tf.log(tf.sub(1, a.square()).add(1e-6));
    const logProb = logProbGauss.sub(correction).sum(-1, true);
    return { a, logProb };
  }

  // Choose an action in ENVIRONMENT units, e.g. [throttle 0..1, steer -1..1]
  act(state, deterministic = false) {
    const norm = tf.tidy(() => {
      const s = tf.tensor2d([state]);
      const a = deterministic ? tf.tanh(this.actor.apply(s)[0]) : this._sample(s).a;
      return Array.from(a.dataSync());
    });
    return this._toEnv(norm);
  }

  // a is in ENVIRONMENT units; stored normalised to [-1, 1]
  remember(s, a, r, s2, done) {
    this.buffer.add(s, this._toNorm(a), r, s2, done);
  }

  _concat(s, a) { return tf.concat([s, a], 1); }

  train() {
    if (this.buffer.size < this.batchSize) return null;
    const batch = this.buffer.sample(this.batchSize);
    const stats = {};

    // --- Critic update ---
    const y = tf.tidy(() => {
      const { a: a2, logProb: lp2 } = this._sample(batch.s2);
      const sa2 = this._concat(batch.s2, a2);
      const qT = tf.minimum(this.q1T.apply(sa2), this.q2T.apply(sa2));
      const alpha = tf.exp(this.logAlpha);
      const soft = qT.sub(alpha.mul(lp2));
      return batch.r.add(tf.scalar(this.gamma).mul(tf.sub(1, batch.d)).mul(soft));
    });

    const criticVars = [...this.q1.trainableWeights, ...this.q2.trainableWeights].map(w => w.val);
    const cLoss = this.criticOpt.minimize(() => {
      const sa = this._concat(batch.s, batch.a);
      const l1 = tf.losses.meanSquaredError(y, this.q1.apply(sa));
      const l2 = tf.losses.meanSquaredError(y, this.q2.apply(sa));
      return l1.add(l2);
    }, true, criticVars);
    stats.criticLoss = cLoss.dataSync()[0];
    cLoss.dispose();
    y.dispose();

    // --- Actor update ---
    const actorVars = this.actor.trainableWeights.map(w => w.val);
    const alphaConst = Math.exp(this.logAlpha.dataSync()[0]); // constant for this step
    let logProbMean = 0;
    const aLoss = this.actorOpt.minimize(() => {
      const { a, logProb } = this._sample(batch.s);
      const sa = this._concat(batch.s, a);
      const q = tf.minimum(this.q1.apply(sa), this.q2.apply(sa));
      logProbMean = logProb.mean().dataSync()[0];
      return logProb.mul(alphaConst).sub(q).mean();
    }, true, actorVars);
    stats.actorLoss = aLoss.dataSync()[0];
    aLoss.dispose();

    // --- Temperature (alpha) update ---
    const target = this.targetEntropy;
    const alphaLoss = this.alphaOpt.minimize(() => {
      return tf.exp(this.logAlpha).mul(-(logProbMean + target));
    }, true, [this.logAlpha]);
    stats.alphaLoss = alphaLoss.dataSync()[0];
    alphaLoss.dispose();
    stats.alpha = Math.exp(this.logAlpha.dataSync()[0]);

    // --- Polyak-average target critics ---
    this._softUpdate(this.q1, this.q1T);
    this._softUpdate(this.q2, this.q2T);

    Object.values(batch).forEach(t => t.dispose());
    return stats;
  }

  _softUpdate(src, tgt) {
    tf.tidy(() => {
      const s = src.getWeights();
      const t = tgt.getWeights();
      tgt.setWeights(s.map((w, i) => w.mul(this.tau).add(t[i].mul(1 - this.tau))));
    });
  }

  async save(name = 'sac-actor') {
    await this.actor.save(`indexeddb://${name}`);
  }
}

// trainLoop() was removed: the sketch owns the loop and calls act/remember/train once per sim step.