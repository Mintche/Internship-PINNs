//
// Physical tags :
//   Surface MILIEU  : 1
//   Surfaces DEFAUT1 : 2
//   Surfaces DEFAUT2 : 3
//   Bord gauche     : 11
//   Bord droit      : 12
//   Bord haut       : 13
//   Bord bas        : 14

Mesh.MshFileVersion = 2.2;
SetFactory("OpenCASCADE");

// -----------------------------------------------------------------------------
// Parametres geometriques
// -----------------------------------------------------------------------------
Lx = 2.0; // domaine en x : [-Lx/2, Lx/2]
Ly = 0.6; // domaine en y : [0, Ly]

DefineConstant[
  cx1 = {-0.30, Min -Lx/2, Max Lx/2, Step 0.01,
         Name "Defaut 1/Centre x"},
  cy1 = { 0.40, Min 0, Max Ly, Step 0.01,
         Name "Defaut 1/Centre y"},
  r1  = { 0.10, Min 0.001, Max Ly/2, Step 0.005,
         Name "Defaut 1/Rayon"},

  cx2 = { 0.30, Min -Lx/2, Max Lx/2, Step 0.01,
         Name "Defaut 2/Centre x"},
  cy2 = { 0.20, Min 0, Max Ly, Step 0.01,
         Name "Defaut 2/Centre y"},
  r2  = { 0.10, Min 0.001, Max Ly/2, Step 0.005,
         Name "Defaut 2/Rayon"}
];

// Garder chaque disque entierement dans le rectangle et eviter que les deux
// disques se chevauchent : distance(centres) > r1 + r2.

// -----------------------------------------------------------------------------
// Parametres de maillage
// -----------------------------------------------------------------------------
h_bulk   = 0.04*2;
h_defaut = 0.015*2;

Mesh.CharacteristicLengthMin = h_defaut;
Mesh.CharacteristicLengthMax = h_bulk;

// -----------------------------------------------------------------------------
// Geometrie et fragmentation
// -----------------------------------------------------------------------------
Rectangle(1) = {-Lx/2, 0, 0, Lx, Ly};
Disk(2) = {cx1, cy1, 0, r1, r1};
Disk(3) = {cx2, cy2, 0, r2, r2};

fragments[] = BooleanFragments{ Surface{1}; Delete; }{ Surface{2, 3}; Delete; };

// Retrouver les surfaces apres l'operation booleenne.
eps = 1e-6;

sDef1[] = Surface In BoundingBox {
  cx1-r1-eps, cy1-r1-eps, -eps,
  cx1+r1+eps, cy1+r1+eps,  eps
};
sDef2[] = Surface In BoundingBox {
  cx2-r2-eps, cy2-r2-eps, -eps,
  cx2+r2+eps, cy2+r2+eps,  eps
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
// Raffinement progressif autour des deux defauts
// -----------------------------------------------------------------------------
Field[1] = Distance;
Field[1].SurfacesList = {sDef1[]};
Field[1].NumPointsPerCurve = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_defaut;
Field[2].SizeMax = h_bulk;
Field[2].DistMin = r1;
Field[2].DistMax = 3*r1;

Field[3] = Distance;
Field[3].SurfacesList = {sDef2[]};
Field[3].NumPointsPerCurve = 100;

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_defaut;
Field[4].SizeMax = h_bulk;
Field[4].DistMin = r2;
Field[4].DistMax = 3*r2;

Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

// -----------------------------------------------------------------------------
// Groupes physiques
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
