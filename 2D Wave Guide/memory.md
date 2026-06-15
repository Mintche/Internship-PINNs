# Mémoire du Projet PINN FWI

## Architecture Alternative FWI
- L'entraînement a été modifié pour utiliser un schéma alterné inspiré de la FWI classique.
- **Phase Forward** : Mise à jour du réseau de champ `u` (`params_uv`) avec la loss de l'équation aux dérivées partielles (PDE) et les conditions aux limites (BC). Le réseau de slowness `m` est gelé.
- **Phase Inversion** : Mise à jour du réseau de slowness `m` (`layers_m`) avec la loss des données (data loss) aux bords et la régularisation de variation totale (TV). Le réseau de champ `u` est gelé via `jax.lax.stop_gradient`.
- La fréquence est fixée à 200 Hz pour capturer une carte probable initiale.
- La taille du réseau `m` a été augmentée pour améliorer sa capacité de représentation (ex: `[2, 128, 128, 128, 1]`).
- Le facteur `c0**2` dans la TV loss a été conservé selon le retour de l'utilisateur.

## Points d'attention futurs
- Ajustement du nombre d'itérations par cycle si nécessaire (actuellement 10 cycles, 1000 iter forward, 500 iter inverse).
- Introduction potentielle du multi-fréquence progressif (bass -> aigu).
