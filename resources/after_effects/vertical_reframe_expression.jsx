// Akio Studio — 16:9 → 9:16 auto-reframe expression
// Apply to: Position of the nested "16_9_Master" precomp layer inside the
// 1080x1920 vertical comp.
//
// Corrections vs. the original architecture spec:
//  1. The original called targetNull.toComp([0,0,0]) and used the result as if
//     it were a coordinate in the *vertical* comp. toComp() resolves into the
//     null's own comp (16_9_Master), so the offset was computed in master-comp
//     pixels and applied unscaled in vertical-comp pixels.
//  2. The original clamp bounds [compWidth - scaledWidth, 0] are only valid
//     when the layer's anchor point sits at its top-left corner. AE's default
//     anchor is the layer center, so the clamp allowed the frame to slide off
//     the canvas. The bounds below are derived from the anchor point, so they
//     hold for any anchor.
//  3. The focal offset is multiplied by the layer's scale so it survives the
//     ~177.8% blow-up needed for a 1080p master to fill a 1920px-tall canvas
//     (or ~88.9% for a UHD master).
//
// Setup: keep "Focal_Point_Null" tracked/keyframed on the subject inside
// 16_9_Master. Scale this nested layer so its height fills the vertical comp
// (uniform scale = vertical comp height / master height * 100).

var master = comp("16_9_Master");
var focal = master.layer("Focal_Point_Null");
// Null position in MASTER-comp pixels (toComp handles parented/animated
// nulls). Sampled at the master-comp time that corresponds to *this* comp's
// current time: startTime offsets are compensated. NOTE: time-stretch and
// time-remap on the nested layer are NOT compensated here — if you retime
// the nested master, fold the same retime into masterTime below.
var masterTime = time - thisLayer.startTime;
var focalX = focal.toComp(focal.transform.anchorPoint, masterTime)[0];

var sx = transform.scale[0] / 100;      // this layer's x scale factor
var ax = transform.anchorPoint[0];      // this layer's anchor, in layer px
var cw = thisComp.width;                // 1080

// x-position that lands the focal point on the vertical center line:
var desired = cw / 2 - (focalX - ax) * sx;

// Coverage clamp — left edge must stay <= 0, right edge >= cw:
//   left edge  = pos - ax * sx
//   right edge = pos + (master.width - ax) * sx
var minX = cw - (master.width - ax) * sx;
var maxX = ax * sx;

// Optional stabilisation: replace `desired` with a temporally smoothed value,
// e.g. smooth(width = 0.4, samples = 9) applied via a helper null, to stop
// micro-jitter from frame-level tracking noise.
[clamp(desired, Math.min(minX, maxX), Math.max(minX, maxX)), value[1]];
