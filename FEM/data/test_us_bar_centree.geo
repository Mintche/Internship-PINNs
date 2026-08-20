// test_ultrasound_barre.geo
// Domaine rectangulaire SYMETRIQUE [-Lx/2, Lx/2] + inclusion barre horizontale
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
Lx = 2.0;    // largeur du rectangle (de -1 à +1)
Ly = 0.6;    // hauteur du rectangle (de 0 à 0.6)

cx = 0;   
cy = Ly/2;

// Paramètres du défaut (barre horizontale)
h_bar = 0.6; // Épaisseur (hauteur) de la barre
w_bar = 0.4; // Longueur de la barre (à ajuster selon tes besoins)

// Calcul du point de départ (coin inférieur gauche) du rectangle-défaut
x_bar = cx - (w_bar / 2);
y_bar = cy - (h_bar / 2);

// ----------------------
// Paramètres de maillage
// ----------------------
h_bulk   = 0.01;   // taille dans le milieu
h_defaut = 0.005;  // taille près du défaut

Mesh.CharacteristicLengthMin = h_defaut;
Mesh.CharacteristicLengthMax = h_bulk;

// ----------------------
// Géométrie
// ----------------------
// Guide d'onde
Rectangle(1) = {-Lx/2, 0, 0, Lx, Ly};          

// Défaut (Barre)
Rectangle(2) = {x_bar, y_bar, 0, w_bar, h_bar};          

// On découpe le guide d'onde par la barre
out[] = BooleanFragments{ Surface{1}; Delete; }{ Surface{2}; Delete; };

// ----------------------
// Récupération des IDs (Mise à jour des coordonnées de recherche)
// ----------------------
eps = 1e-6;

// Surface du défaut : on englobe le rectangle de la barre
sDef[] = Surface In BoundingBox {x_bar-eps, y_bar-eps, -1, x_bar+w_bar+eps, y_bar+h_bar+eps, 1};

// Surface totale : couvre tout de -Lx/2 à +Lx/2
sAll[] = Surface In BoundingBox {-Lx/2-eps, -eps, -1, Lx/2+eps, Ly+eps, 1};

// On déduit la surface du milieu (sAll - sDef)
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
// Les distances ci-dessous remplacent le rayon (r et 3*r) 
// pour créer le dégradé de taille de maille autour du rectangle.
Field[2].DistMin = 0.05; 
Field[2].DistMax = 0.25; 

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