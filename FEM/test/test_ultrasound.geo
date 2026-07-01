// test_ultrasound.geo
// Domaine rectangulaire + inclusion circulaire (defaut)
// Physical tags:
//  - Surface MILIEU : 1
//  - Surface DEFAUT : 2
//  - SIGMA_PLUS_L (gauche) : 11
//  - SIGMA_MINUS_L (droite): 12
//  - BORD_HAUT : 13
//  - BORD_BAS  : 14

Mesh.MshFileVersion = 2.2;

SetFactory("OpenCASCADE");

// ----------------------
// Paramètres géométriques
// ----------------------
Lx = 1.0;    // largeur du rectangle
Ly = 0.6;    // hauteur du rectangle

cx = 0.5;    // centre du défaut
cy = 0.3;
r  = 0.08;   // rayon du défaut

// ----------------------
// Paramètres de maillage
// ----------------------
h_bulk   = 0.04;   // taille dans le milieu
h_defaut = 0.015;  // taille près du défaut

Mesh.CharacteristicLengthMin = h_defaut;
Mesh.CharacteristicLengthMax = h_bulk;

// ----------------------
// Géométrie
// ----------------------
Rectangle(1) = {0, 0, 0, Lx, Ly};          // Surface 1 (provisoire)
Disk(2)      = {cx, cy, 0, r, r};          // Surface 2 (provisoire)

// On découpe le rectangle par le disque:
// - le rectangle devient l'extérieur (milieu) avec un trou
// - le disque est la surface du défaut
out[] = BooleanFragments{ Surface{1}; Delete; }{ Surface{2}; Delete; };

// Après BooleanFragments, les IDs changent.
// On récupère les surfaces résultantes par localisation (simple et robuste).
//  - surface défaut : celle qui contient le point (cx,cy)
//  - surface milieu : l'autre
sDef[] = Surface In BoundingBox {cx-r-1e-6, cy-r-1e-6, -1, cx+r+1e-6, cy+r+1e-6, 1};
sAll[] = Surface In BoundingBox {0-1e-6, 0-1e-6, -1, Lx+1e-6, Ly+1e-6, 1};

sMil[] = sAll[];
// Retire sDef de sMil (si jamais il y a plusieurs surfaces dans sDef, on gère quand même)
For i In {0:#sDef[]-1}
  sMil[] -= {sDef[i]};
EndFor

// ----------------------
// Champs de taille (raffinement près du défaut)
// ----------------------
Field[1] = Distance;
Field[1].SurfacesList = {sDef[]};
Field[1].NumPointsPerCurve = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_defaut;
Field[2].SizeMax = h_bulk;
Field[2].DistMin = r;
Field[2].DistMax = 3*r;

Background Field = 2;

// ----------------------
// Physical groups (surfaces)
// ----------------------
Physical Surface(1) = {sMil[]}; // MILIEU
Physical Surface(2) = {sDef[]}; // DEFAUT

// ----------------------
// Physical groups (bords)
// ----------------------
// On récupère les courbes du contour du rectangle via BoundingBox.
// Gmsh ne garantit pas l'ordre des courbes, donc on les classe par position.

eps = 1e-6;

// Courbes sur x=0 (gauche)
cLeft[]  = Curve In BoundingBox {-eps, -eps, -1, eps, Ly+eps, 1};
// Courbes sur x=Lx (droite)
cRight[] = Curve In BoundingBox {Lx-eps, -eps, -1, Lx+eps, Ly+eps, 1};
// Courbes sur y=Ly (haut)
cTop[]   = Curve In BoundingBox {-eps, Ly-eps, -1, Lx+eps, Ly+eps, 1};
// Courbes sur y=0 (bas)
cBot[]   = Curve In BoundingBox {-eps, -eps, -1, Lx+eps, eps, 1};

Physical Curve(11) = {cLeft[]};   // SIGMA_PLUS_L
Physical Curve(12) = {cRight[]};  // SIGMA_MINUS_L
Physical Curve(13) = {cTop[]};    // BORD_HAUT
Physical Curve(14) = {cBot[]};    // BORD_BAS

// Optionnel : forcer triangles
Mesh.RecombineAll = 0;
