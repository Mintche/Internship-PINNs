// test_ultrasound_sym.geo
// Domaine rectangulaire SYMETRIQUE [-Lx/2, Lx/2] + inclusion circulaire
// Physical tags:
//  - Surface MILIEU : 1
//  - Surface DEFAUT : 2
//  - SIGMA_GAUCHE (x=-Lx/2) : 11
//  - SIGMA_DROITE (x=+Lx/2) : 12
//  - BORD_HAUT    (y=Ly)    : 13
//  - BORD_BAS     (y=0)     : 14

Mesh.MshFileVersion = 2.2;
SetFactory("OpenCASCADE");

// ----------------------
// Paramètres géométriques
// ----------------------
Lx = 4.0;    // largeur du rectangle
Ly = 0.6;    // hauteur du rectangle

cx = -0.2;  
cy = 0.2;
r  = 0.1;   // rayon du défaut

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
// MODIFICATION ICI : Le rectangle est centré en X
// Rectangle(1) = {X_start, Y_start, Z_start, Width, Height}
Rectangle(1) = {-Lx/2, 0, 0, Lx, Ly};          

Disk(2)      = {cx, cy, 0, r, r};          

// On découpe le rectangle par le disque
out[] = BooleanFragments{ Surface{1}; Delete; }{ Surface{2}; Delete; };

// ----------------------
// Récupération des IDs (Mise à jour des coordonnées de recherche)
// ----------------------
eps = 1e-6;

// Surface du défaut : autour de (cx, cy)
sDef[] = Surface In BoundingBox {cx-r-eps, cy-r-eps, -1, cx+r+eps, cy+r+eps, 1};

// Surface totale : couvre tout de -Lx/2 à +Lx/2
sAll[] = Surface In BoundingBox {-Lx/2-eps, -eps, -1, Lx/2+eps, Ly+eps, 1};

sMil[] = sAll[];
For i In {0:#sDef[]-1}
  sMil[] -= {sDef[i]};
EndFor

// ----------------------
// Champs de taille (Distance au défaut)
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

// GAUCHE : x = -Lx/2
cLeft[]  = Curve In BoundingBox {-Lx/2-eps, -eps, -1, -Lx/2+eps, Ly+eps, 1};

// DROITE : x = +Lx/2
cRight[] = Curve In BoundingBox {Lx/2-eps, -eps, -1, Lx/2+eps, Ly+eps, 1};

// HAUT : y = Ly (couvre tout X de -Lx/2 à Lx/2)
cTop[]   = Curve In BoundingBox {-Lx/2-eps, Ly-eps, -1, Lx/2+eps, Ly+eps, 1};

// BAS : y = 0 (couvre tout X de -Lx/2 à Lx/2)
cBot[]   = Curve In BoundingBox {-Lx/2-eps, -eps, -1, Lx/2+eps, eps, 1};

Physical Curve(11) = {cLeft[]};   // SIGMA_GAUCHE
Physical Curve(12) = {cRight[]};  // SIGMA_DROITE
Physical Curve(13) = {cTop[]};    // BORD_HAUT
Physical Curve(14) = {cBot[]};    // BORD_BAS

Mesh.RecombineAll = 0;