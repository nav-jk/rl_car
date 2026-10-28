// Soft Actor-Critic (continuous actions) in TensorFlow.js
// Browser: <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs"></script>  (load BEFORE this file)

const tf = typeof window !== 'undefined' ? window.tf : require('@tensorflow/tfjs');

// ---------- (de)serialisation helpers ----------
// Tensors are stored as base64 of their raw bytes, so a save file is lossless and reasonably compact.
function bytesToB64(u8) {
  let bin = '';
  for (let i = 0; i < u8.length; i += 0x8000) bin += String.fromCharCode.apply(null, u8.subarray(i, i + 0x8000));
  return btoa(bin);
}
function b64ToBytes(b64) {
  const bin = atob(b64);
  const u8 = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return u8;
}
function f32ToB64(f32) { return bytesToB64(new Uint8Array(f32.buffer, f32.byteOffset, f32.byteLength)); }
function b64ToF32(b64) { const u8 = b64ToBytes(b64); return new Float32Array(u8.buffer, u8.byteOffset, u8.byteLength / 4); }

function tensorToObj(t) {
  const d = t.dataSync();   // Float32Array or Int32Array
  return { dtype: t.dtype, shape: t.shape, data: bytesToB64(new Uint8Array(d.buffer, d.byteOffset, d.byteLength)) };
}
function objToTensor(o) {
  const u8 = b64ToBytes(o.data);
  const arr = o.dtype === 'int32' ? new Int32Array(u8.buffer) : new Float32Array(u8.buffer);
  return tf.tensor(arr, o.shape, o.dtype);
}

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

  // Flat Float32 rows: [s..., a..., r, s2..., d]  (storage order and write pointer are kept)
  exportState(stateDim, actionDim) {
    const w = stateDim * 2 + actionDim + 2, n = this.data.length;
    const out = new Float32Array(n * w);
    for (let i = 0; i < n; i++) {
      const it = this.data[i];
      out.set([...it.s, ...it.a, it.r, ...it.s2, it.d], i * w);
    }
    return { n, idx: this.idx, width: w, data: f32ToB64(out) };
  }
  importState(o, stateDim, actionDim) {
    const w = stateDim * 2 + actionDim + 2;
    if (o.width !== w) throw new Error('replay buffer row width mismatch');
    const f = b64ToF32(o.data);
    this.data = [];
    for (let i = 0; i < o.n; i++) {
      const r = Array.from(f.subarray(i * w, (i + 1) * w));
      this.data.push({
        s: r.slice(0, stateDim),
        a: r.slice(stateDim, stateDim + actionDim),
        r: r[stateDim + actionDim],
        s2: r.slice(stateDim + actionDim + 1, 2 * stateDim + actionDim + 1),
        d: r[w - 1],
      });
    }
    this.idx = o.idx % this.capacity;
  }

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
    this.hidden = hidden;
    this.lr = opts.lr ?? 3e-4;
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

  // ---------- full training-state snapshot ----------
  // Everything needed to resume training exactly: all 5 networks, temperature, Adam moments,
  // and (optionally, it is the big part) the replay buffer.
  async exportState({ includeBuffer = true } = {}) {
    const nets = m => m.getWeights().map(tensorToObj);
    const opt = async o => (await o.getWeights()).map(w => ({ name: w.name, ...tensorToObj(w.tensor) }));
    return {
      format: 'sac-state', version: 1,
      config: {
        stateDim: this.stateDim, actionDim: this.actionDim, hidden: this.hidden,
        gamma: this.gamma, tau: this.tau, batchSize: this.batchSize, lr: this.lr,
        targetEntropy: this.targetEntropy, actionLow: this.actionLow, actionHigh: this.actionHigh,
      },
      weights: {
        actor: nets(this.actor), q1: nets(this.q1), q2: nets(this.q2),
        q1T: nets(this.q1T), q2T: nets(this.q2T),
        logAlpha: this.logAlpha.dataSync()[0],
      },
      optimizers: { actor: await opt(this.actorOpt), critic: await opt(this.criticOpt), alpha: await opt(this.alphaOpt) },
      buffer: includeBuffer ? this.buffer.exportState(this.stateDim, this.actionDim) : null,
    };
  }

  // Restores in place (the SAC object, and the arrays the sketch holds, stay valid).
  // Returns a list of warnings (e.g. optimizer state could not be restored).
  async importState(st) {
    if (st.format !== 'sac-state') throw new Error('not a SAC state');
    const c = st.config, warnings = [];
    if (c.stateDim !== this.stateDim || c.actionDim !== this.actionDim || c.hidden !== this.hidden)
      throw new Error(`model shape mismatch: file has state ${c.stateDim}, action ${c.actionDim}, hidden ${c.hidden}; ` +
        `this agent has state ${this.stateDim}, action ${this.actionDim}, hidden ${this.hidden}`);

    const setNet = (m, arr) => {
      const ts = arr.map(objToTensor);
      m.setWeights(ts);
      ts.forEach(t => t.dispose());
    };
    setNet(this.actor, st.weights.actor); setNet(this.q1, st.weights.q1); setNet(this.q2, st.weights.q2);
    setNet(this.q1T, st.weights.q1T); setNet(this.q2T, st.weights.q2T);
    this.logAlpha.assign(tf.scalar(st.weights.logAlpha));

    const setOpt = async (o, arr, label) => {
      try {
        await o.setWeights(arr.map(w => ({ name: w.name, tensor: objToTensor(w) })));
      } catch (e) {
        warnings.push(`${label} optimizer state not restored (${e.message}); Adam moments restart from zero`);
      }
    };
    await setOpt(this.actorOpt, st.optimizers.actor, 'actor');
    await setOpt(this.criticOpt, st.optimizers.critic, 'critic');
    await setOpt(this.alphaOpt, st.optimizers.alpha, 'alpha');

    if (st.buffer) this.buffer.importState(st.buffer, this.stateDim, this.actionDim);
    else { this.buffer.data = []; this.buffer.idx = 0; warnings.push('no replay buffer in file: starting with an empty one'); }
    return warnings;
  }

  async save(name = 'sac-actor') {
    await this.actor.save(`indexeddb://${name}`);
  }
}

// trainLoop() was removed: the sketch owns the loop and calls act/remember/train once per sim step.