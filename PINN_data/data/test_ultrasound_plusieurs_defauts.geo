// test_ultrasound_multi_defects_list.geo
// Tube rectangle + plusieurs défauts (liste fixée, sans Rand())
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
Lx = 1.0;
Ly = 0.6;

// ----------------------
// Paramètres de maillage
// ----------------------
h_bulk   = 0.045;
h_defaut = 0.015;

Mesh.CharacteristicLengthMin = h_defaut;
Mesh.CharacteristicLengthMax = h_bulk;

// ----------------------
// Domaine
// ----------------------
Rectangle(1) = {0, 0, 0, Lx, Ly};

// ----------------------
// Liste des défauts (centres + rayons)
// -> Tous les disques doivent rester dans [0,Lx]x[0,Ly]
// -> Évite d’être trop près des bords, sinon tu “manges” la frontière.
// ----------------------
Disk(101) = {0.18, 0.3, 0, 0.022, 0.022};
Disk(102) = {0.79, 0.4, 0, 0.025, 0.025};
//Disk(103) = {0.5, 0.15, 0, 0.03, 0.03};

surfsDef[] = {101, 102};

// ----------------------
// Découpe du rectangle par tous les disques
// ----------------------
out[] = BooleanFragments{ Surface{1}; Delete; }{ Surface{surfsDef[]}; Delete; };

// ----------------------
// Récupérer surfaces MILIEU / DEFAUT
// (Méthode : bounding box autour de chaque disque)
// ----------------------
sAll[] = Surface In BoundingBox {0-1e-6, 0-1e-6, -1, Lx+1e-6, Ly+1e-6, 1};
sDefAll[] = {};

// Pour chaque disque (cx,cy,r), on prend la surface dans sa bounding box.
sOne[] = Surface In BoundingBox {0.18-0.022-1e-6, 0.3-0.022-1e-6, -1, 0.18+0.022+1e-6, 0.3+0.022+1e-6, 1}; sDefAll[] += {sOne[]};
sOne[] = Surface In BoundingBox {0.79-0.025-1e-6, 0.4-0.025-1e-6, -1, 0.79+0.025+1e-6, 0.4+0.025+1e-6, 1}; sDefAll[] += {sOne[]};
//sOne[] = Surface In BoundingBox {0.5-0.03-1e-6, 0.15-0.03-1e-6, -1, 0.5+0.03+1e-6, 0.15+0.03+1e-6, 1}; sDefAll[] += {sOne[]};

Unique(sDefAll[]);

// sMil = sAll - sDefAll
sMil[] = sAll[];
For i In {0:#sDefAll[]-1}
  sMil[] -= {sDefAll[i]};
EndFor
Unique(sMil[]);

// ----------------------
// Raffinement près des défauts
// ----------------------
Field[1] = Distance;
Field[1].SurfacesList = {sDefAll[]};
Field[1].NumPointsPerCurve = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_defaut;
Field[2].SizeMax = h_bulk;
// on prend des distances cohérentes avec les rayons ci-dessus
Field[2].DistMin = 0.018;
Field[2].DistMax = 0.090;

Background Field = 2;

// ----------------------
// Physical groups (surfaces)
// ----------------------
Physical Surface(1) = {sMil[]};    // MILIEU
Physical Surface(2) = {sDefAll[]}; // DEFAUT

// ----------------------
// Physical groups (bords)
// ----------------------
eps = 1e-6;

cLeft[]  = Curve In BoundingBox {-eps, -eps, -1, eps, Ly+eps, 1};
cRight[] = Curve In BoundingBox {Lx-eps, -eps, -1, Lx+eps, Ly+eps, 1};
cTop[]   = Curve In BoundingBox {-eps, Ly-eps, -1, Lx+eps, Ly+eps, 1};
cBot[]   = Curve In BoundingBox {-eps, -eps, -1, Lx+eps, eps, 1};

Physical Curve(11) = {cLeft[]};
Physical Curve(12) = {cRight[]};
Physical Curve(13) = {cTop[]};
Physical Curve(14) = {cBot[]};

Mesh.RecombineAll = 0;
