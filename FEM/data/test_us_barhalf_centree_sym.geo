// Rectangular waveguide with two symmetric half-bar defects using different material tags.
//
// Physical tags:
//   Surface MILIEU      : 1
//   Barre basse DEFAUT 1 : 2
//   Barre haute DEFAUT 2 : 3
//   Bord gauche         : 11
//   Bord droit          : 12
//   Bord haut           : 13
//   Bord bas            : 14

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
h_bar = 0.3; // Épaisseur (hauteur) de la barre
w_bar = 0.4; // Longueur de la barre (à ajuster selon tes besoins)

// Points de départ (coin inférieur gauche) des deux rectangles-défauts
x_bar = cx - (w_bar / 2);
y_bar_bottom = 0;
y_bar_top = cy;

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

// Défaut 1 (barre basse)
Rectangle(2) = {x_bar, y_bar_bottom, 0, w_bar, h_bar};

// Défaut 2 (barre haute)
Rectangle(3) = {x_bar, y_bar_top, 0, w_bar, h_bar};

// On découpe le guide d'onde par les deux barres
fragments[] = BooleanFragments{ Surface{1}; Delete; }{ Surface{2, 3}; Delete; };

// ----------------------
// Récupération des IDs (Mise à jour des coordonnées de recherche)
// ----------------------
eps = 1e-6;

// Surfaces des défauts : on englobe chaque rectangle de barre
sDefBottom[] = Surface In BoundingBox {
  x_bar-eps, y_bar_bottom-eps, -eps,
  x_bar+w_bar+eps, y_bar_bottom+h_bar+eps, eps
};
sDefTop[] = Surface In BoundingBox {
  x_bar-eps, y_bar_top-eps, -eps,
  x_bar+w_bar+eps, y_bar_top+h_bar+eps, eps
};

// Surface totale : couvre tout de -Lx/2 à +Lx/2
sAll[] = Surface In BoundingBox {
  -Lx/2-eps, -eps, -eps,
   Lx/2+eps, Ly+eps, eps
};

// On déduit la surface du milieu (sAll - sDefBottom - sDefTop)
sMil[] = sAll[];
For i In {0:#sDefBottom[]-1}
  sMil[] -= {sDefBottom[i]};
EndFor
For i In {0:#sDefTop[]-1}
  sMil[] -= {sDefTop[i]};
EndFor

// ----------------------
// Champs de taille (Distance au défaut)
// ----------------------
Field[1] = Distance;
Field[1].SurfacesList = {sDefBottom[]};
Field[1].NumPointsPerCurve = 100;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_defaut;
Field[2].SizeMax = h_bulk;
// Les distances ci-dessous remplacent le rayon (r et 3*r) 
// pour créer le dégradé de taille de maille autour du rectangle.
Field[2].DistMin = 0.05; 
Field[2].DistMax = 0.25; 

Field[3] = Distance;
Field[3].SurfacesList = {sDefTop[]};
Field[3].NumPointsPerCurve = 100;

Field[4] = Threshold;
Field[4].InField = 3;
Field[4].SizeMin = h_defaut;
Field[4].SizeMax = h_bulk;
Field[4].DistMin = 0.05;
Field[4].DistMax = 0.25;

Field[5] = Min;
Field[5].FieldsList = {2, 4};
Background Field = 5;

// ----------------------
// Physical groups (surfaces)
// ----------------------
Physical Surface(1) = {sMil[]}; // MILIEU
Physical Surface(2) = {sDefBottom[]}; // DEFAUT BAS
Physical Surface(3) = {sDefTop[]};    // DEFAUT HAUT

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
