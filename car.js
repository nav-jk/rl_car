class Car {
  constructor(x, y) {
    this.pos = createVector(x, y)
    this.vel = createVector(0, 0)
    this.acc = createVector(0, 0)
    this.facing = createVector(0, -1);

    this.width = 15
    this.height = 25
    this.color = color('red')

    this.drag = 0.07

    this.min_vel = 0.1
  }


  show() {
    push()
    fill(this.color)
    translate(this.pos.x, this.pos.y)
    rotate(PI/2 + this.facing.heading())
    strokeWeight(1)
    // set the point of turning around the middle front of the vehicle
    rect(-this.width / 2, -this.height / 4, this.width, this.height)
    strokeWeight(5)
    point(0, 0)
    pop()
  }
  update() {
    this.vel.add(this.acc)
    this.pos.add(this.vel)

    this.acc.mult(0)
  }

  envForces() {
    this.applyForce(p5.Vector.mult(this.vel, -this.drag))
  }

  applyForce(f) {
    this.acc.add(f)
  }
  drive(throttle) {
    let f = p5.Vector.mult(this.facing, (throttle));
    this.applyForce(f)
  }
  steer(da) {
    this.facing.rotate(da*this.vel.mag()) // might be wrong
    this.vel.setHeading(this.facing.heading())
  }

  driveByUser() {
    if(keyIsDown(UP_ARROW)) {
      this.drive(0.1)
    }
    if(keyIsDown(RIGHT_ARROW)) {
      this.steer(+0.02)
    }
    else if (keyIsDown(LEFT_ARROW)) {
      this.steer(-0.02)
    }
    if (keyIsDown(DOWN_ARROW)) {
      this.drive(-0.1)
    }
  }
  get boundaryCoords() {
    // points with respect to turn point
    // apply rotation
    // render and see to verify
    let p1 = createVector(- this.width / 2, - this.height / 4)
    let p2 = createVector(this.width / 2, - this.height / 4)
    let p3 = createVector(this.width / 2, 3/4 * this.height)
    let p4 = createVector(- this.width / 2, 3/4 * this.height)

    p1.rotate(PI/2 + this.facing.heading())
    p2.rotate(PI/2 +this.facing.heading())
    p3.rotate(PI/2 +this.facing.heading())
    p4.rotate(PI/2 +this.facing.heading())

    let res = [p1, p2, p3, p4]

    for (let v of res) {
      v.add(this.pos)
    }

    return res;
  }
}
