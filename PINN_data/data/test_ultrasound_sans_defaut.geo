// test_ultrasound_nodefect_sym.geo
// Domaine rectangulaire SYMETRIQUE en X : [-Lx/2, +Lx/2]
// Y reste défini sur [0, Ly]

Mesh.MshFileVersion = 2.2;
SetFactory("OpenCASCADE");

// ----------------------
// Paramètres géométriques
// ----------------------
Lx = 1.0;    // largeur totale
Ly = 0.6;    // hauteur totale
h_bulk = 0.04;   

Mesh.CharacteristicLengthMin = h_bulk;
Mesh.CharacteristicLengthMax = h_bulk;

// ----------------------
// Géométrie
// ----------------------
// Rectangle(1) = {X_coin, Y_coin, Z_coin, Largeur, Hauteur};
// On décale le X du coin gauche à -Lx/2
Rectangle(1) = {-Lx/2, 0, 0, Lx, Ly};   

// ----------------------
// Physical groups (surfaces)
// ----------------------
Physical Surface(1) = {1}; // MILIEU

// ----------------------
// Physical groups (bords)
// ----------------------
eps = 1e-6;

// GAUCHE : x = -Lx/2
// On cherche dans une boite très fine autour de -Lx/2
cLeft[]  = Curve In BoundingBox {-Lx/2-eps, -eps, -1, -Lx/2+eps, Ly+eps, 1};

// DROITE : x = +Lx/2
// On cherche dans une boite très fine autour de +Lx/2
cRight[] = Curve In BoundingBox {Lx/2-eps, -eps, -1, Lx/2+eps, Ly+eps, 1};

// HAUT : y = Ly (couvre tout x de -Lx/2 à Lx/2)
cTop[]   = Curve In BoundingBox {-Lx/2-eps, Ly-eps, -1, Lx/2+eps, Ly+eps, 1};

// BAS : y = 0 (couvre tout x de -Lx/2 à Lx/2)
cBot[]   = Curve In BoundingBox {-Lx/2-eps, -eps, -1, Lx/2+eps, eps, 1};

Physical Curve(11) = {cLeft[]};   // SIGMA_GAUCHE
Physical Curve(12) = {cRight[]};  // SIGMA_DROITE
Physical Curve(13) = {cTop[]};    // BORD_HAUT
Physical Curve(14) = {cBot[]};    // BORD_BAS

Mesh.RecombineAll = 0;