// ---- tuning constants ----
const CAR_MAX_THROTTLE = 0.15;  // force at throttle = 1  (terminal speed = force / drag = 0.15 / 0.07 ~ 2.1)
const CAR_MAX_STEER = 0.03;     // steer = 1 -> rotation per frame = 0.03 * speed (radians)
const CAR_MAX_SPEED = 2.2;      // used only to normalise speed into [0, 1]

const PROGRESS_SCALE = 10;      // reward per radian travelled around the track (full lap ~ 63)
const TIME_PENALTY = 0.002;     // tiny cost per step so standing still is never free
const CRASH_PENALTY = 10;
const LAP_BONUS = 50;

function wrapAngle(a) {
  while (a > Math.PI) a -= 2 * Math.PI;
  while (a < -Math.PI) a += 2 * Math.PI;
  return a;
}
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

class Driver {
  constructor(car, track = null) {
    this.car = car;
    this.track = null;
    this.center = { x: 0, y: 0 };

    this.sensor_count = 8;
    this.sensor_range = 100;
    this.sensors = [];

    this.passingLap = false;
    this.lap_count = 0;
    this.prevAngle = 0;

    if (track) this.setTrack(track);
  }

  setTrack(track) {
    this.track = track;
    // centre of the inner wall = the point we measure progress around
    const p = track.inner_pts;
    this.center = {
      x: p.reduce((s, q) => s + q[0], 0) / p.length,
      y: p.reduce((s, q) => s + q[1], 0) / p.length,
    };
  }

  sensorInit() {
    this.sensors = [];
    for (let j = 0; j < this.sensor_count; j++) {
      this.sensors.push(new Sensor(this.car.pos.x, this.car.pos.y, this.sensor_range,
        (2 * PI / this.sensor_count) * j, this.track));
    }
  }

  // ---------- progress around the track ----------
  // Angle of the car around the track centre. The track is driven "up the left side",
  // i.e. clockwise on screen, which is the direction of INCREASING atan2 angle (y points down).
  get angle() {
    return Math.atan2(this.car.pos.y - this.center.y, this.car.pos.x - this.center.x);
  }

  resetProgress() { this.prevAngle = this.angle; }

  // car heading relative to the direction of travel along the track, in (-PI, PI]
  // 0 = pointing along the track, +-PI = pointing backwards
  get relativeHeading() {
    const tangent = this.angle + Math.PI / 2;
    return wrapAngle(this.car.facing.heading() - tangent);
  }

  // ---------- observation ----------
  // All values are numbers in [-1, 1]:
  //   8 sensor distances (0 = touching, 1 = nothing within range)
  //   speed / max speed
  //   cos and sin of heading relative to the track direction
  get state() {
    const s = this.sensors.map(sn => sn.normDist);
    s.push(clamp(this.car.vel.mag() / CAR_MAX_SPEED, 0, 1));
    const rel = this.relativeHeading;
    s.push(Math.cos(rel), Math.sin(rel));
    return s;
  }

  // ---------- reward ----------
  // Call exactly ONCE per simulation step (it updates prevAngle).
  reward(crashed, lapDone) {
    const a = this.angle;
    const progress = wrapAngle(a - this.prevAngle);   // > 0 forward, < 0 backward
    this.prevAngle = a;

    let r = progress * PROGRESS_SCALE - TIME_PENALTY;
    if (crashed) r -= CRASH_PENALTY;
    if (lapDone) r += LAP_BONUS;
    return r;
  }

  isOffTrack() {
    return this.track.checkLeftTrackLimits(this.car.boundaryCoords);
  }

  // ---------- lap detection ----------
  segmentsIntersect(p1, p2, p3, p4) {
    const o = (a, b, c) => (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    const d1 = o(p3, p4, p1), d2 = o(p3, p4, p2);
    const d3 = o(p1, p2, p3), d4 = o(p1, p2, p4);
    return d1 * d2 < 0 && d3 * d4 < 0;
  }

  // Has side effects (passingLap, lap_count): call once per step.
  hasCompletedLap() {
    const a = { x: 10, y: 235 };
    const b = { x: 70, y: 235 };

    const pts = this.car.boundaryCoords;
    let intersecting = false;
    for (let i = 0; i < pts.length; i++) {
      if (this.segmentsIntersect(pts[i], pts[(i + 1) % pts.length], a, b)) {
        intersecting = true;
        break;
      }
    }

    // The line points +x. cross < 0 means the car is heading up (-y) on the left side,
    // which is the driving direction of this track.
    const h = this.car.facing;
    const cross = (b.x - a.x) * h.y - (b.y - a.y) * h.x;
    const correctDirection = cross < 0;

    if (intersecting) {
      this.passingLap = correctDirection;
      return false;
    }
    if (this.passingLap) {
      this.passingLap = false;
      if (correctDirection) {
        this.lap_count++;
        return true;
      }
    }
    return false;
  }

  // ---------- control ----------
  // thrust in [0, 1], steer in [-1, 1]
  drive(thrust, steer) {
    this.car.steer(clamp(steer, -1, 1) * CAR_MAX_STEER);
    this.car.drive(clamp(thrust, 0, 1) * CAR_MAX_THROTTLE);
  }

  // move sensors with the car and read them. Call after car.update().
  update() {
    for (const s of this.sensors) {
      s.start = this.car.pos;
      s.rotate(this.car.facing.heading());
      s.update();
    }
  }

  show() {
    for (const s of this.sensors) s.show();
    // lap line
    push();
    stroke('green');
    line(10, 235, 70, 235);
    pop();
    text(this.lap_count, 500, 500);
  }
}


class Sensor {
  constructor(x, y, l, heading, track) {
    this.start = createVector(x, y);
    this.range = l;
    this.heading = createVector(0, -1); // points up
    this.heading.setMag(l);
    this.def_heading = heading;
    this.heading.setHeading(heading);
    this.track = track;
    this.sense_dist_sqr = Number.POSITIVE_INFINITY;
    this.hit = null;
  }

  rotate(angle) {
    this.heading.setHeading(this.def_heading + angle);
  }

  get end() {
    return p5.Vector.add(this.start, this.heading);
  }

  // 0 = touching, 1 = nothing in range. Always a finite number.
  get normDist() {
    if (!Number.isFinite(this.sense_dist_sqr)) return 1;
    return Math.min(1, Math.sqrt(this.sense_dist_sqr) / this.range);
  }

  show() {
    line(this.start.x, this.start.y, this.end.x, this.end.y);
    if (this.hit) {
      push();
      strokeWeight(5);
      stroke('red');
      point(this.hit.x, this.hit.y);
      pop();
    }
  }

  sense(x1, y1, x2, y2) {
    const isect = [NaN, NaN];
    if (segseg(isect, [this.start.x, this.start.y], [this.end.x, this.end.y], [x1, y1], [x2, y2])) {
      return { x: isect[0], y: isect[1] };
    }
    return false;
  }

  // nearest wall hit this frame (starts from Infinity every time,
  // the old version started from last frame's value so distances could only shrink)
  senseObstacles() {
    let min = Number.POSITIVE_INFINITY;
    this.hit = null;
    for (const ob of this.track.obstacles) {
      const res = this.sense(ob.start.x, ob.start.y, ob.end.x, ob.end.y);
      if (res) {
        const d = (res.x - this.start.x) ** 2 + (res.y - this.start.y) ** 2;
        if (d < min) {
          min = d;
          this.hit = res;
        }
      }
    }
    this.sense_dist_sqr = min;
  }

  update() {
    this.senseObstacles();
  }
}