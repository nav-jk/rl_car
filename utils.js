function intersectionRectLine(pts, p1, p2) {
  // form line equation
  for (let i = 0; i < pts.length;i++) {
    let ps = pts[i % pts.length ];
    let pe = pts[(i+1) % pts.length];
    if (intersect(ps.x, ps.y, pe.x, pe.y, p1.x, p1.y, p2.x, p2.y)) {
      return true;
    }
  }

  return false;
}

function intersect(x1, y1, x2, y2, x3, y3, x4, y4) {
  // Check if none of the lines are of length 0
  if ((x1 === x2 && y1 === y2) || (x3 === x4 && y3 === y4)) {
    return false;
  }

  denominator = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1);


  let ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3));
  let ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) ;


  // is the intersection along the segments
  if (ua < 0 || ua > 1 || ub < 0 || ub > 1) {
    console.log(ua, ub)
    return false;
  }

  // Return a object with the x and y coordinates of the intersection
  let x = x1 + ua * (x2 - x1);
  let y = y1 + ua * (y2 - y1);

  return { x, y };
}


// from https://gist.github.com/mattdesl/47412d930dcd8cd765c871a65532ffac

function sqr (x) {
  return x * x
}


function dist2 (v, w) {
  return sqr(v[0] - w[0]) + sqr(v[1] - w[1])
}


// p - point
// v - start point of segment
// w - end point of segment
function distToSegmentSquared (p, v, w) {
  const l2 = dist2(v, w)
  if (l2 === 0)
      return dist2(p, v)

  let t = ((p[0] - v[0]) * (w[0] - v[0]) + (p[1] - v[1]) * (w[1] - v[1])) / l2
  t = Math.max(0, Math.min(1, t))
  return dist2(p, [ v[0] + t * (w[0] - v[0]), v[1] + t * (w[1] - v[1]) ])
}


// p - point
// t0 - start point of segment
// t1 - end point of segment
// epsilon - maximum distance from the segment that a point can still be considered on the segment
// return boolean indicating if p is on the segment
function distToSegment (p, t0, t1, epsilon) {
  return !(Math.sqrt(distToSegmentSquared(p, t0, t1)) > epsilon)
}



const DONT_INTERSECT = 0
const DO_INTERSECT = 1
const COLINEAR = 2


/*  Ported from Mukesh Prasad's public domain code:
 *    http://www.realtimerendering.com/resources/GraphicsGems/gemsii/xlines.c
 *
 *   This function computes whether two line segments,
 *   respectively joining the input points (x1,y1) -- (x2,y2)
 *   and the input points (x3,y3) -- (x4,y4) intersect.
 *   If the lines intersect, the return value is an array
 *   containing coordinates of the point of intersection.
 *
 *   Params
 *        p1, p2   Coordinates of endpoints of one segment.
 *        p3, p4   Coordinates of endpoints of other segment.
 *
 *   The value returned by the function is an enumeration of DONT_INTERSECT | DO_INTERSECT | COLINEAR
 */

function _segseg (out, p1, p2, p3, p4) {
  let x1 = p1[0]
  let y1 = p1[1]
  let x2 = p2[0]
  let y2 = p2[1]
  let x3 = p3[0]
  let y3 = p3[1]
  let x4 = p4[0]
  let y4 = p4[1]

  let a1, a2, b1, b2, c1, c2 // Coefficients of line eqns.
  let r1, r2, r3, r4         // 'Sign' values
  let denom, offset          // Intermediate values
  let x, y                   // Intermediate return values

  // Compute a1, b1, c1, where line joining points 1 and 2
  // is "a1 x  +  b1 y  +  c1  =  0".
  a1 = y2 - y1
  b1 = x1 - x2
  c1 = x2 * y1 - x1 * y2

  // Compute r3 and r4.
  r3 = a1 * x3 + b1 * y3 + c1
  r4 = a1 * x4 + b1 * y4 + c1

  // Check signs of r3 and r4.  If both point 3 and point 4 lie on
  // same side of line 1, the line segments do not intersect.
  if ( r3 !== 0 && r4 !== 0 && ((r3 >= 0 && r4 >= 0) || (r3 < 0 && r4 < 0)))
    return DONT_INTERSECT

  // Compute a2, b2, c2
  a2 = y4 - y3
  b2 = x3 - x4
  c2 = x4 * y3 - x3 * y4

  // Compute r1 and r2
  r1 = a2 * x1 + b2 * y1 + c2
  r2 = a2 * x2 + b2 * y2 + c2

  // Check signs of r1 and r2.  If both point 1 and point 2 lie
  // on same side of second line segment, the line segments do
  // not intersect.
  if (r1 !== 0 && r2 !== 0 && ((r1 >= 0 && r2 >= 0) || (r1 < 0 && r2 < 0)))
    return DONT_INTERSECT

  // Line segments intersect: compute intersection point.
  denom = a1 * b2 - a2 * b1

  if (denom === 0)
    return COLINEAR

  offset = denom < 0 ? - denom / 2 : denom / 2

  x = b1 * c2 - b2 * c1
  y = a2 * c1 - a1 * c2

  out[0] = ( x < 0 ? x : x ) / denom
  out[1] = ( y < 0 ? y : y ) / denom
  
  return DO_INTERSECT
}


function segseg (out, p1, p2, p3, p4, epsilon=0) {
  const result = _segseg(out, p1, p2, p3, p4)

  if (result === DO_INTERSECT)
    return result

  // handle colinear cases and when a line segment endpoint lies on the other segment
  if (distToSegment(p1, p3, p4, epsilon)) {
    out[0] = p1[0]
    out[1] = p1[1]
    return true
  }

  if (distToSegment(p2, p3, p4, epsilon)) {
    out[0] = p2[0]
    out[1] = p2[1]
    return true
  }

  if (distToSegment(p3, p1, p2, epsilon)) {
    out[0] = p3[0]
    out[1] = p3[1]
    return true
  }

  if (distToSegment(p4, p1, p2, epsilon)) {
    out[0] = p4[0]
    out[1] = p4[1]
    return true
  }

  return false
}


function onSegment(x1, y1, x2, y2, x, y) {
    let c = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1);

    if (c !== 0)
        return false;

    return Math.min(x1, x2) <= x && x <= Math.max(x1, x2) &&
           Math.min(y1, y2) <= y && y <= Math.max(y1, y2);
}

function isInside(arr, x, y) {
    let n = arr.length;
    let inside = false;

    for (let i = 0, j = n - 1; i < n; j = i++) {
        let [x1, y1] = arr[i];
        let [x2, y2] = arr[j];

        // Point lies on the current edge
        if (onSegment(x1, y1, x2, y2, x, y))
            return true;

        // Check whether the horizontal ray from (x, y)
        // intersects the current edge
        let intersect =
            ((y1 > y) !== (y2 > y)) &&
            (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1);

        if (intersect)
            inside = !inside;
    }

    return inside;
}
