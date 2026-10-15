let car, track, driver, pauseBtn, paused=false;

function setup() {
  createCanvas(windowWidth, windowHeight);

  car = new Car(30, 200)
  track = new Track()

  driver = new Driver(car);

  driver.setTrack(track);
  driver.sensorInit(); // only after set track
  pauseBtn = createButton("Pause")
  pauseBtn.position(0, 100);
  pauseBtn.mousePressed(() => paused = !paused)
}

function draw() {

  if (!paused) {
    background(220);
    track.show()
    driver.show()
    driver.update()
    driver.sense()

    car.show()
    car.update()
    car.envForces()
    car.driveByUser()
  }
  // car.getBoundaryCoords()
  //
}



class Track {
  constructor() {
    this.outer_pts = [[10, 10], [470, 10], [470, 470], [10, 470]];
    this.inner_pts = [[70, 70], [70, 400], [400, 400], [400, 70]];
    this.out_vertices = this.outer_pts.length;
    this.in_vertices = this.inner_pts.length;

    this.obstacles = []
    this.checkpoints = []

    // init
    for (let i = 0; i < this.in_vertices;i++){
      this.addStaticObstacle(this.inner_pts[i][0], this.inner_pts[i][1], this.inner_pts[(i + 1) % this.in_vertices][0], this.inner_pts[(i+1) % this.in_vertices][1]);
    }
    for (let i = 0; i < this.out_vertices;i++){
      this.addStaticObstacle(this.outer_pts[i][0], this.outer_pts[i][1], this.outer_pts[(i + 1) % this.out_vertices][0], this.outer_pts[(i+1) % this.out_vertices][1])
    }
  }
  addStaticObstacle(x1, y1, x2, y2) {
    // assuming obstacles are static
    this.obstacles.push({
      start: {
        x: x1,
        y: y1
      },
      end: {
        x: x2,
        y: y2
      },
      intersect: {
        x: NaN,
        y: NaN
      },
      isIntersecting: false,
      intersectDistSqr: -1
    })

  }

  show() {
    beginShape();

  // Exterior vertices, clockwise winding.

  for (let i of this.outer_pts) {
    vertex(i[0], i[1])
  }
  

  // Interior vertices, counter-clockwise winding.
  beginContour();
  for (let i of this.inner_pts) {
    vertex(i[0], i[1])
  }
  endContour(CLOSE);

  // Stop drawing the shape.
  endShape(CLOSE);
  }

  checkLeftTrackLimits(pts) {
    // check if rectangle is inside track
    for (let p of pts) {
      if (isInside(this.inner_pts, p.x, p.y)) return true;
      if (!isInside(this.outer_pts, p.x, p.y)) return true;
    }
    return false

  }
}
