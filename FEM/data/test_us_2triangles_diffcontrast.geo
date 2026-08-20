// Rectangular waveguide with two triangular defects using different material tags.
//
// Physical tags:
//   Surface MILIEU      : 1
//   Triangle DEFAUT 1   : 2
//   Triangle DEFAUT 2   : 3
//   Bord gauche         : 11
//   Bord droit          : 12
//   Bord haut           : 13
//   Bord bas            : 14

Mesh.MshFileVersion = 2.2;
SetFactory("OpenCASCADE");

// -----------------------------------------------------------------------------
// Geometry parameters
// -----------------------------------------------------------------------------
Lx = 2.0; // domain in x: [-Lx/2, Lx/2]
Ly = 0.6; // domain in y: [0, Ly]

DefineConstant[
  t1x1 = {-0.25, Min -Lx/2, Max Lx/2, Step 0.01, Name "Triangle 1/x1"},
  t1y1 = {Ly, Min 0,     Max Ly,   Step 0.01, Name "Triangle 1/y1"},
  t1x2 = {0.25, Min -Lx/2, Max Lx/2, Step 0.01, Name "Triangle 1/x2"},
  t1y2 = {Ly, Min 0,     Max Ly,   Step 0.01, Name "Triangle 1/y2"},
  t1x3 = {0.0, Min -Lx/2, Max Lx/2, Step 0.01, Name "Triangle 1/x3"},
  t1y3 = {0.3, Min 0,     Max Ly,   Step 0.01, Name "Triangle 1/y3"},

  t2x1 = {-0.25, Min -Lx/2, Max Lx/2, Step 0.01, Name "Triangle 2/x1"},
  t2y1 = { 0.0, Min 0,     Max Ly,   Step 0.01, Name "Triangle 2/y1"},
  t2x2 = { 0.25, Min -Lx/2, Max Lx/2, Step 0.01, Name "Triangle 2/x2"},
  t2y2 = { 0.0, Min 0,     Max Ly,   Step 0.01, Name "Triangle 2/y2"},
  t2x3 = { 0.0, Min -Lx/2, Max Lx/2, Step 0.01, Name "Triangle 2/x3"},
  t2y3 = { 0.3, Min 0,     Max Ly,   Step 0.01, Name "Triangle 2/y3"}
];

// Keep each triangle strictly inside the rectangle and avoid overlapping the
// two triangles. The FEM generator assigns contrast by physical surface tag.

// -----------------------------------------------------------------------------
// Mesh parameters
// -----------------------------------------------------------------------------
h_bulk = 0.01;
h_defaut = 0.005;

Mesh.CharacteristicLengthMin = h_defaut;
Mesh.CharacteristicLengthMax = h_bulk;

// -----------------------------------------------------------------------------
// Geometry and fragmentation
// -----------------------------------------------------------------------------
Rectangle(1) = {-Lx/2, 0, 0, Lx, Ly};

Point(101) = {t1x1, t1y1, 0, h_defaut};
Point(102) = {t1x2, t1y2, 0, h_defaut};
Point(103) = {t1x3, t1y3, 0, h_defaut};
Line(101) = {101, 102};
Line(102) = {102, 103};
Line(103) = {103, 101};
Curve Loop(101) = {101, 102, 103};
Plane Surface(2) = {101};

Point(201) = {t2x1, t2y1, 0, h_defaut};
Point(202) = {t2x2, t2y2, 0, h_defaut};
Point(203) = {t2x3, t2y3, 0, h_defaut};
Line(201) = {201, 202};
Line(202) = {202, 203};
Line(203) = {203, 201};
Curve Loop(201) = {201, 202, 203};
Plane Surface(3) = {201};

fragments[] = BooleanFragments{ Surface{1}; Delete; }{ Surface{2, 3}; Delete; };

// Recover surfaces after BooleanFragments.
eps = 1e-6;

t1xmin = Min(t1x1, Min(t1x2, t1x3));
t1xmax = Max(t1x1, Max(t1x2, t1x3));
t1ymin = Min(t1y1, Min(t1y2, t1y3));
t1ymax = Max(t1y1, Max(t1y2, t1y3));

t2xmin = Min(t2x1, Min(t2x2, t2x3));
t2xmax = Max(t2x1, Max(t2x2, t2x3));
t2ymin = Min(t2y1, Min(t2y2, t2y3));
t2ymax = Max(t2y1, Max(t2y2, t2y3));

sDef1[] = Surface In BoundingBox {
  t1xmin-eps, t1ymin-eps, -eps,
  t1xmax+eps, t1ymax+eps,  eps
};
sDef2[] = Surface In BoundingBox {
  t2xmin-eps, t2ymin-eps, -eps,
  t2xmax+eps, t2ymax+eps,  eps
};
sAll[] = Surface In BoundingBox {
  -Lx/2-eps, -eps, -eps,
   Lx/2+eps, Ly+eps, eps
};

sMil[] = sAll[];
For i In {0:#sDef1[]-1}
  sMil[] -= {sDef1[i]};
EndFor
For i In {0:#sDef2[]-1}
  sMil[] -= {sDef2[i]};
EndFor

// -----------------------------------------------------------------------------
// Progressive refinement around both triangular defects
// -----------------------------------------------------------------------------
Field[1] = Distance;
Field[1].SurfacesList = {sDef1[]};
Field[1].NumPointsPerCurve = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_defaut;
Field[2].SizeMax = h_bulk;
Field[2].DistMin = 0.04;
Field[2].DistMax = 0.16;

Field[3] = Distance;
Field[3].SurfacesList = {sDef2[]};
Field[3].NumPointsPerCurve = 100;

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_defaut;
Field[4].SizeMax = h_bulk;
Field[4].DistMin = 0.04;
Field[4].DistMax = 0.16;

Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

// -----------------------------------------------------------------------------
// Physical groups
// -----------------------------------------------------------------------------
Physical Surface(1) = {sMil[]};
Physical Surface(2) = {sDef1[]};
Physical Surface(3) = {sDef2[]};

cLeft[] = Curve In BoundingBox {
  -Lx/2-eps, -eps, -eps, -Lx/2+eps, Ly+eps, eps
};
cRight[] = Curve In BoundingBox {
   Lx/2-eps, -eps, -eps,  Lx/2+eps, Ly+eps, eps
};
cTop[] = Curve In BoundingBox {
  -Lx/2-eps, Ly-eps, -eps, Lx/2+eps, Ly+eps, eps
};
cBot[] = Curve In BoundingBox {
  -Lx/2-eps, -eps, -eps, Lx/2+eps, eps, eps
};

Physical Curve(11) = {cLeft[]};
Physical Curve(12) = {cRight[]};
Physical Curve(13) = {cTop[]};
Physical Curve(14) = {cBot[]};

Mesh.RecombineAll = 0;
