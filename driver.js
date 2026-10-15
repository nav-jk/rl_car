class Driver {
  constructor(car, track=null) {
    this.car = car;
    this.track = track;


    this.sensor_count = 8;
    this.sensor_range = 100;
    this.sensors = []
}

  setTrack(track) {
    this.track = track;
  }

  sensorInit() {
    // sensor init
    for (let j = 0; j < this.sensor_count;j++) {
      this.sensors.push(new Sensor(this.car.pos.x,this.car.pos.y, this.sensor_range, (2*PI/this.sensor_count)*j, this.track))
      
    }
  }
  show() {
    for (let s of this.sensors) {
      s.show(intersect=true)
    }
  }

  drive() {
    
  }

  sense() {
    // draw sensor lines
    // check intersection and consider as state
    if (this.track.checkLeftTrackLimits(this.car.boundaryCoords)) {
      // give penalty
    }
  }

  applyPolicy() {

  }

  update() {
    for (let s of this.sensors) {
      s.start = this.car.pos
      s.rotate(this.car.facing.heading())
    }
    this.updatePolicy()
    this.sensors.forEach(s => s.update()) 
  }

  updatePolicy() {
  }
}


class Sensor {
  constructor(x, y, l, heading, track) {
    this.start = createVector(x, y)
    this.heading = createVector(0, -1); // points up
    this.heading.setMag(l);
    this.def_heading = heading;
    this.heading.setHeading(heading);
    this.track = track;
  }

  
  rotate(angle) {
    this.heading.setHeading(this.def_heading + angle)
  }

  get end() {
    return  p5.Vector.add(this.start, this.heading)
  }
  show() {
    line(this.start.x, this.start.y, this.end.x, this.end.y)
  }

  sense(x1, y1, x2, y2) {
    const isect = [ NaN, NaN ]
    if (segseg(isect, [ this.start.x, this.start.y ], [ this.end.x, this.end.y ],  [ x1, y1 ], [ x2, y2 ])) {
      return {
        x: isect[0],
        y: isect[1]
      };
    }
    return false
    // return intersect(this.start.x, -this.start.y, this.end.x, -this.end.y, x1, -y1, x2, -y2)
  }
  senseObstacles(show=false) {
    for (let ob of this.track.obstacles) {
      let res = this.sense(ob.start.x, ob.start.y, ob.end.x, ob.end.y)

      if(show=true){
        // stroke('blue')
        // line(ob.start.x, ob.start.y, ob.end.x, ob.end.y)
        // stroke('black')
      }
      if (res) {
        ob.intersect = res;
        ob.isIntersecting = true;
        ob.intersectDistSqr = sq(res.x - this.start.x) + sq(res.y - this.start.y);
        if (show) {
          strokeWeight(5)
          stroke('red')
          point(res.x, res.y)
          strokeWeight(1)
          stroke('black')
        }
      }
      else {
        ob.isIntersecting = false;
        ob.intersectDistSqr = Number.POSITIVE_INFINITY;
      }
    }
  }
  update(){  
    this.senseObstacles(true)
  }
}
