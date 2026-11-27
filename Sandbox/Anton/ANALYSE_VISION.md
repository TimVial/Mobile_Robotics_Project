# Vision : repérer la carte, redresser la vue, localiser le Thymio

Analyse des options offertes par le cours MICRO-452, puis recommandation argumentée pour notre projet.

Sources (les numéros renvoient aux slides imprimées) :
**V** = *Slides 02_03 - Perception, Vision and Machine Learning*, **L** = *Slides 06 - Localisation*,
**C** = *Slides 01 - Components of a Mobile Robot*, **P** = *Project Presentation*, **Pièges** = *Project Common Pitfalls*,
**T** = notebook *Exercises/Computer Vision Tutorials/OpenCV Examples.ipynb*, **E2** = exercices 2 (parties 1 et 2),
**CS2**, **CS8** = études de cas 2 et 8.

---

## En bref

> **Recommandation : des marqueurs ArUco partout (4 coins, robot, but) sur une carte blanche et mate. Au démarrage, on calcule une fois une homographie qui convertit l'image en millimètres de la carte. Pour le robot, on corrige en plus la parallaxe due à la hauteur de son marqueur.**
>
> - Le détecteur ArUco et la formule de l'homographie vont un peu plus loin que le cours, mais s'expliquent entièrement avec ses notions (§4.2, §8.5).
> - Dans notre simulation, ArUco trouve le robot dans 100 % des images sous tous les éclairages testés. Les pastilles couleur actuelles tombent à 0 % dans une image 2,5× plus sombre, et à 37 % sous l'ombre d'une personne.
> - La correction de parallaxe fait passer l'erreur de position d'environ 10 mm (jusqu'à 2 cm aux coins) à environ 1–3 mm.
> - Le point faible d'ArUco est le flou de mouvement. Le filtre de Kalman comble les images où le robot n'est pas vu.
>
> Montage et logiciel : §8. Chiffres : §7.

---

## 1. Ce que la vision doit fournir

| Quand | Sortie | Utilisée par |
|---|---|---|
| Une fois, au démarrage | homographie image → carte (mm), position de la caméra, polygones des obstacles, position du but | navigation globale |
| À chaque image | pose $z = (x, y, \theta)$ du **centre de l'axe des roues**, en mm et rad, dans le repère de la carte (origine en bas à gauche, y vers le haut, θ anti-horaire depuis x), ou `None` si le robot n'est pas vu | `kalman.update` |
| Une fois, mesurée | covariance de mesure $R$ | `CAMERA_VARIANCES` dans `kalman.py`, aujourd'hui « assumed » |

Pendant la démo, le jury peut **cacher la caméra**, **déplacer le robot (kidnapping)** et **changer le départ et le but** (P5, P13). La vision doit donc :

1. **ne jamais donner une fausse pose** : mieux vaut `None`, et le Kalman continue avec l'odométrie ;
2. **retrouver le robot n'importe où, en une seule image**, sans initialisation ni suivi qui puisse décrocher ;
3. **tenir sous l'éclairage de la salle**, avec des gens autour de la table qui font de l'ombre.

## 2. Ce que le cours met à disposition

| Outil du cours | Où | Utile pour |
|---|---|---|
| Modèle sténopé $x = fX/Z$, $y = fY/Z$. Avec une caméra, impossible de retrouver P depuis p « without a priori knowledge » | V11–V13 | redresser la vue. Notre connaissance a priori : tout est sur le sol, d'où l'homographie (§4) |
| Tâches 3D en robotique mobile : *camera calibration* (distorsions), *camera localisation (extrinsic calibration)*, *object pose estimation* | V6, V8 | nos trois problèmes, dans cet ordre |
| « Known 2D patterns… a good feature needs to be **easy to extract and easy to match** » | V8 | choisir des motifs artificiels faits pour être détectés |
| Filtres : moyenne, gaussien, médian, bilatéral | V15–16, T §3.1, E2 | débruiter avant seuillage et contours |
| Seuillage global, adaptatif (moyenne locale − C), Otsu ; égalisation d'histogramme et CLAHE | T §3.2–3.3 | segmenter malgré les variations de lumière |
| Contours (Sobel, Canny), transformée de Hough | V17–21, E2 | bord de la carte, lignes |
| Morphologie, contours, centroïde par les moments, détection de blobs | T §4, E2 | pastilles, obstacles |
| Pattern matching (SSD, corrélation) et ses limites en rotation et en échelle | V22–23, T §5, E2 | gabarit du robot ou des coins |
| Points d'intérêt (Harris, FAST, SIFT…), invariants à l'échelle et à la rotation | V24–29 | carte texturée |
| Choix de l'espace couleur ; discriminant linéaire (LDA) pour séparer deux couleurs | Pièges, V38–44 | marqueurs colorés |
| Suivi meanShift sur histogramme de teinte | T §5 « Tracking » | suivre le robot |
| Réseaux de neurones et CNN (« black box ») | V51–56 | détection d'objets |
| Balises (*beacons*) = signaux de **position précisément connue** ; la vision comme capteur de position absolue | L8, L11, C30 | coins de la carte, robot |
| Erreurs systématiques (« distortion caused by the optics of a camera »), aléatoires (« hue instability of camera »), flou de mouvement | C28 | quoi corriger, quoi craindre |
| Angle d'un robot vu par la caméra d'un drone, grâce à une ligne noire sur son dessus | CS2 | cap du robot |
| Navette localisée par « the camera detecting the markers » | CS8 | marqueurs et filtre |
| « Do not hesitate to use libraries… make sure you understand how the libraries and the underlying algorithms work » | P15, Pièges | OpenCV et ArUco sont permis, à condition de savoir les expliquer |

**Ce que le cours ne traite pas explicitement** : les QR codes, les marqueurs fiduciaires (ArUco, AprilTag), la formule de l'homographie, le modèle de distorsion et `solvePnP`. Ce sont de petits pas au-delà du cours, construits sur les notions du tableau.

## 3. Problème 1 : repérer la carte

But : trouver dans l'image au moins 4 points dont on connaît la position sur la carte, **et savoir lequel est lequel**. Sinon le repère peut tourner de 90°.

### A. Contour du carré entier (code actuel : `Project/vision.py`, `Sandbox/Anton`)
Gris, flou gaussien, Canny, plus grand contour, `approxPolyDP` jusqu'à 4 sommets, puis `order_points` (tri par somme et différence des coordonnées).
- **Pour** : simple, entièrement dans le cours (V17, T §3.4), déjà codé. Coins précis quand tout va bien (0,7 px en simulation).
- **Contre** :
  - Il suffit d'un coin caché (main, câble, robot au bord) ou d'un obstacle qui touche le bord pour ne plus avoir 4 sommets. On perd alors la carte : 0 % en simulation avec une main sur un coin.
  - Les seuils de Canny sont fixes : 0 % de réussite en simulation dans une image 4× plus sombre.
  - **Ambiguïté** : un carré a 4 symétries. `order_points` choisit le coin « haut-gauche » d'après l'image : si la caméra est tournée de plus de 45°, l'origine change de coin. Vers 45° pile, le tri devient instable.
  - Le plus grand contour peut être autre chose que la carte : bord de la table, feuille, ombre.

### B. Bordure noire + transformée de Hough
Ruban noir autour de la carte, Canny, Hough (E2 partie 1), 4 droites. Les coins sont les intersections.
- **Pour** : très « cours ». Une droite partiellement cachée est quand même trouvée, grâce au vote. Des coins obtenus par intersection de droites sont précis.
- **Contre** : il faut choisir les 4 bonnes droites parmi celles des obstacles et des deux bords du ruban, et régler l'accumulateur. Même ambiguïté de 90° qu'en A.

### C. Pastilles de couleur aux 4 coins
Seuillage couleur (HSV, Lab, ou la LDA du cours, V38), blobs, centroïdes, homographie. Une couleur différente pour l'origine lève l'ambiguïté.
- **Pour** : simple, centroïdes précis, dans le cours (T §4, E2).
- **Contre** :
  - C'est le piège vision n°1 du cours : « Vision – don't choose the right color space ». La couleur mesurée dépend de l'éclairage et de la balance des blancs automatique de la caméra (« hue instability », C28).
  - Tout objet de la même couleur peut créer un faux coin.
  - Avec un coin caché, il ne reste que 3 points : on peut seulement calculer une transformation affine, qui ne corrige plus la perspective.

### D. Gabarit (template matching) des coins
`matchTemplate` d'un motif imprimé (V22–23, T §5).
- **Contre** : comme le dit le tutoriel du cours, ce n'est invariant ni en rotation ni en échelle (T §5). Il faudrait connaître d'avance l'orientation et la taille de la carte dans l'image. **Écarté.**

### E. Points d'intérêt (SIFT, ORB) sur une carte texturée
On imprime une texture riche, on apparie ses points avec une image de référence, puis on calcule l'homographie avec RANSAC (V24–29).
- **Pour** : invariant à la rotation, à l'échelle et à la perspective ; supporte une occlusion partielle.
- **Contre** :
  - La texture gêne la détection des obstacles, qui demande un fond uni.
  - C'est plus lent, et moins précis et moins stable que des coins dessinés pour être détectés.
  - Difficile à expliquer en détail à l'oral. **Disproportionné** pour notre besoin.

### F. Marqueurs fiduciaires aux 4 coins (ArUco ; et les QR codes ?)
Un **marqueur ArUco** est un carré noir qui contient une grille de bits : 4×4 bits pour le dictionnaire `DICT_4X4_50`. C'est une **balise de position connue** (L8), conçue pour être « easy to extract and easy to match » (V8). Pour chaque marqueur, OpenCV renvoie son **identifiant** et ses **4 coins dans un ordre fixe**, avec une précision sous-pixel.
- **Pour** :
  - **Pas d'ambiguïté** : grâce à l'identifiant, on sait quel coin est l'origine, quelle que soit la rotation de la caméra.
  - 4 marqueurs × 4 coins = **16 points**, donc une homographie aux moindres carrés. Avec 3 marqueurs (12 points), ça marche encore.
  - **Rien à régler** : la détection travaille en niveaux de gris, avec un seuil adaptatif puis un code binaire, sans couleur. En simulation, les 4 coins sont trouvés sous tous les éclairages, même dans une image 4× plus sombre.
  - **Pas de faux positifs** : un faux marqueur devrait avoir une bordure noire, une grille et un code du dictionnaire. Aucun faux identifiant sur 300 images simulées.
- **Contre** :
  - Il faut une marge blanche autour de chaque marqueur et du papier mat : un reflet blanc efface les bits.
  - Le marqueur entier doit être visible.
  - La méthode est hors cours, mais elle s'explique avec le cours (§8.5).

**Et les QR codes ?** Le cours n'en parle pas, et ce n'est pas le bon outil.
- Un QR code sert à **stocker des données** : il a au moins 21×21 modules, plus 4 modules de marge.
- Un marqueur ArUco 4×4 n'a que 6×6 modules, bordure comprise.
- À taille imprimée égale, les modules ArUco sont donc **3,5× plus gros**. Il est détecté à plus faible résolution, supporte mieux le flou, est plus rapide à décoder, et ses coins sont plus précis.

En résumé, ArUco, c'est le QR code fait pour localiser des robots.

**AprilTag** (`DICT_APRILTAG_36h11`, aussi disponible dans OpenCV) rejette encore mieux les faux positifs. Mais ses modules (8×8) sont 25 % plus petits : à notre résolution, le dictionnaire 4×4 est le meilleur compromis.

### G. Clic manuel des 4 coins au démarrage
Avec `cv2.setMouseCallback`, on clique les 4 coins dans l'ordre.
- **Pour** : fiable à 100 % tant que la caméra ne bouge pas ; aucun réglage.
- **Contre** : pas automatique, et à refaire si quelqu'un touche la caméra. **Bon plan B.**

## 4. Problème 2 : calibrer la vue (corriger l'angle)

### 4.1 Pourquoi une échelle unique (`MM_PER_PIXEL`) ne suffit pas
Le code actuel suppose une caméra parfaitement perpendiculaire au sol.
- Dès qu'elle penche, le bord proche de la carte paraît plus grand que le bord éloigné : la carte devient un trapèze (V13).
- Avec 5° d'inclinaison à 1,6 m de haut, l'échelle varie d'environ 5 % d'un bord à l'autre. Cela donne environ ±1,5 cm d'erreur aux bords d'une carte de 1 m.
- De plus, la carte est en général tournée dans l'image.

### 4.2 L'homographie : la projection perspective du cours, appliquée à un plan
Pour un point du sol $(X, Y, 0)$, le modèle sténopé (V12) s'écrit, à un facteur $s$ près :

$$
s \begin{pmatrix}u\\ v\\ 1\end{pmatrix}
= K \begin{bmatrix} r_1 & r_2 & r_3 & t \end{bmatrix} \begin{pmatrix}X\\ Y\\ 0\\ 1\end{pmatrix}
= \underbrace{K \begin{bmatrix} r_1 & r_2 & t \end{bmatrix}}_{H\ (3\times 3)} \begin{pmatrix}X\\ Y\\ 1\end{pmatrix}
$$

Comme $Z = 0$, la colonne $r_3$ disparaît. Entre le plan du sol et l'image, la projection est donc une **homographie** $H$ : une matrice 3×3 définie à un facteur près, soit 8 inconnues. C'est la « connaissance a priori » qui lève l'ambiguïté de profondeur de V12 : on sait que les points sont sur le sol.

- **Combien de points.** Chaque correspondance pixel ↔ point de la carte (mm) donne 2 équations linéaires. 4 points suffisent (`cv2.getPerspectiveTransform`). Avec plus de points, `cv2.findHomography` résout aux moindres carrés, ce qui réduit le bruit.
- **Quelle unité.** On calcule directement $H$ : **image → millimètres de la carte**. Vision, planification, Kalman et contrôle travaillent alors dans la même unité, sans `MM_PER_PIXEL` ni retournement de l'axe y.
- **Ne pas redresser toute l'image à chaque image.** Pour le robot, il suffit de transformer ses 4 coins (`cv2.perspectiveTransform`) : c'est quasi instantané et sans interpolation. La vue de dessus complète (`cv2.warpPerspective`) ne sert qu'au démarrage, pour extraire les obstacles, et pour l'affichage.

### 4.3 La distorsion de l'objectif (calibration intrinsèque, optionnelle)
L'homographie suppose une caméra sténopé parfaite. Or une webcam grand-angle courbe les droites près des bords : c'est l'erreur systématique « distortion caused by the optics » (C28).

**Correction (V6, « camera calibration »)** :
1. Une seule fois, photographier un damier imprimé sous une quinzaine d'angles.
2. `cv2.calibrateCamera` donne la matrice $K$ et les coefficients de distorsion.
3. À chaque image, `cv2.undistortPoints` corrige seulement les coins détectés : une vingtaine de points, coût nul.

**Est-ce nécessaire ?** Deux tests simples :
- une règle droite posée près du bord de l'image doit paraître droite ;
- l'erreur mesurée au centre de la carte (§8.3) doit rester sous 5 mm.

Un natel corrige souvent déjà la distorsion lui-même. Bonus : $K$ sert aussi à localiser la caméra (§4.4).

### 4.4 Le piège : le marqueur du robot n'est pas sur le sol (parallaxe)
L'homographie est celle du **plan du sol**, mais le dessus du Thymio est à $h \approx 5{,}5$ cm. Le rayon de vue qui passe par le marqueur touche donc le sol **plus loin** du point situé sous la caméra (le nadir $N$) :

```
         C          caméra, à la hauteur Hc
         |\
         | \
         |  \       rayon de vue
         |   \
         |    \
         |     •    P  : marqueur, à la hauteur h
         |     :\
  sol ---N-----x-•---   x  : vraie position (sous P)
         |<-d->| |      •  : P', ce que donne l'homographie du sol
         |<-d'-->|      d' = d · Hc / (Hc − h)
```

Par triangles semblables (même raisonnement que $x = fX/Z$, V12) :

$$ P = N + (P' - N)\,\frac{H_c - h}{H_c} $$

**Ordre de grandeur, avec la caméra à 1,65 m** : erreur nulle sous la caméra, **17 à 21 mm aux coins de la carte**, environ 10 mm en médiane.
- C'est la plus grosse erreur de toute la chaîne, et elle touche **toutes** les méthodes de détection du robot, pastilles comprises. Le code actuel ne la corrige pas.
- Elle ne change pas l'angle : entre deux plans parallèles, la projection depuis C est une homothétie de centre N.

Pour la corriger, il faut connaître $N$ et $H_c$ :
- **Au mètre ruban.** Avec 5 cm d'erreur sur N et 10 cm sur $H_c$, l'erreur de parallaxe tombe à environ 2 mm. En simulation, l'erreur totale médiane est alors de 2,6 mm (§7).
- **Par calcul** (V6, « camera localisation (extrinsic calibration) »). `cv2.solvePnP`, avec les 16 coins des marqueurs et $K$, donne la position de la caméra dans le repère de la carte. Même avec une focale fausse de 10 % (par exemple $K$ estimé depuis l'angle de champ annoncé par le fabricant), l'erreur médiane reste sous 3 mm.

**Variante sans calcul** : poser les 4 marqueurs de coin sur des cales de la même hauteur que le marqueur du robot. Mais alors ce sont les obstacles, au sol, qui subissent la parallaxe.

### 4.5 Calibrer une fois, puis surveiller
La caméra et la carte sont fixes, donc $H$ ne change pas. On la calcule **au démarrage, sur environ 30 images** : en moyennant les coins, on divise leur bruit par environ 5. Ensuite, on la fige :
- un coin caché plus tard (une main, le robot qui passe dessus) ne gêne plus rien ;
- la pose du robot n'hérite pas du tremblement d'une homographie recalculée à chaque image ;
- **surveillance** : si les coins visibles s'écartent de plus d'environ 3 px de leur position de calibration pendant environ 1 s, c'est que quelqu'un a touché la caméra. On recalibre alors automatiquement.

## 5. Problème 3 : localiser le robot (x, y, θ)

Le point à mesurer est le **centre de l'axe des roues**, au niveau du trou du stylo : c'est le point dont l'odométrie décrit le mouvement, comme dans `Sandbox/Tim/vision.py`. Le cap θ doit être connu sur 360°, sans confusion entre l'avant et l'arrière.

| | Méthode | Cours | Pour | Contre |
|---|---|---|---|---|
| R1 | Corps blanc du Thymio sur fond sombre (`Sandbox/Tim/vision.py`) : seuil, contour, `minAreaRect`, avant/arrière par l'asymétrie | T §3.3, E2 | rien à coller sur le robot | impose un **fond sombre**, donc des obstacles clairs ; tout objet blanc gêne ; le cap repose sur une petite asymétrie |
| R2 | Deux pastilles verte et bleue (`Project/vision.py`) : seuils HSV, centroïdes, `atan2` | T §4, E2, V38 | simple, et **robuste au flou** de mouvement | **sensible à la lumière** : avec des seuils réglés en bonne lumière, 0 % de détection dans une image 2,5× plus sombre et 37 % sous l'ombre d'une personne (simulation). Les LED du Thymio, un but ou un obstacle de même couleur créent de faux blobs |
| R3 | Une forme colorée asymétrique (flèche, triangle) : centroïde et axe | T §4 | une seule couleur | mêmes défauts que R2 |
| R4 | Gabarit (template matching) du robot | V22, T §5 | entièrement dans le cours | **pas invariant en rotation** (T §5), donc pas d'angle ; il faudrait un gabarit par angle |
| R5 | Suivi meanShift ou CamShift sur histogramme de teinte | T §5 | dans le cours | initialisation manuelle, dérive, **ne retrouve pas le robot après un kidnapping** |
| R6 | Ligne noire sur le dessus, puis Sobel ou Hough (CS2) | V18–21, CS2 | angle précis et peu coûteux | angle connu **à 180° près** : il faut un second indice pour l'avant |
| R7 | Détecteur appris (CNN, YOLO) | V51–56 | robuste en général | jeu de données, entraînement, « black box » ; ne donne pas d'angle précis. Disproportionné |
| R8 | **Marqueur ArUco sur le dessus** | V8, L8, puis au-delà du cours | identifiant unique ; 4 coins sous-pixel, donc position **et** cap sur 360° ; insensible à la lumière ; jamais de fausse pose ; retrouve le robot n'importe où en une image (kidnapping) | sensible au **flou de mouvement** : fiable jusqu'à environ 7 mm de déplacement pendant la pose, presque toujours perdu vers 14 mm. Un marqueur partiellement caché n'est pas vu, et le Kalman prend le relais |

Avec R8, la pose se lit sur les 4 coins transformés en mm :
1. le centre est la moyenne des 4 coins ;
2. la direction « avant » va du milieu du bord bas au milieu du bord haut du marqueur ;
3. on applique la correction de parallaxe (§4.4) ;
4. si le marqueur n'est pas centré sur l'axe des roues, on recule du décalage mesuré.

## 6. Comparaison

Notation : ++ très bon, + bon, 0 moyen, − faible, −− rédhibitoire.

**Carte**

| Option | Lumière | Coin caché | Ambiguïté, faux positifs | Précision | Effort | Proche du cours |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| A. Contour (actuel) | − | −− | − | + | ++ | ++ |
| B. Bordure + Hough | 0 | + | − | + | − | ++ |
| C. Pastilles couleur | −− | − | 0 | + | + | ++ |
| D. Gabarit | 0 | 0 | − | 0 | 0 | ++ |
| E. SIFT / ORB | + | ++ | + | 0 | − | + |
| **F. ArUco** | **++** | **++** | **++** | **++** | **+** | **+** |
| G. Clic manuel | ++ | ++ | ++ | + | ++ | 0 |

**Robot**

| Option | Lumière | Flou | Faux positifs | Cap sur 360° | Kidnapping | Effort | Proche du cours |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| R1. Corps blanc | − | + | − | 0 | + | ++ | ++ |
| R2. Deux pastilles (actuel) | −− | ++ | − | + | + | ++ | ++ |
| R3. Forme colorée | −− | + | − | + | + | + | ++ |
| R4. Gabarit | 0 | 0 | − | −− | + | + | ++ |
| R5. Suivi meanShift | − | + | − | − | −− | + | ++ |
| R6. Ligne + Hough | 0 | 0 | 0 | − | + | 0 | ++ |
| R7. CNN | + | + | + | − | + | −− | − |
| **R8. ArUco** | **++** | **−** | **++** | **++** | **++** | **+** | **+** |

## 7. Banc d'essai (simulation)

Pour vérifier ces affirmations, nous avons simulé la scène en images de synthèse, avec une vérité terrain exacte. Le banc d'essai réimplémente nos méthodes actuelles A et R2 avec leurs paramètres, et les compare à ArUco.

**La scène simulée** :
- caméra sténopé 1280×720, focale 914 px (environ 70° de champ), à 1,65 m de haut, inclinée de 3° et 2,5° ;
- carte blanche de 1 m sur une table brune, 4 marqueurs de coin de 80 mm, obstacles noirs ;
- dessus du Thymio à 55 mm, portant soit un marqueur ArUco de 70 mm, soit une pastille verte (avant) et une bleue (arrière) comme dans `Project/vision.py`.

**Les méthodes comparées** :
- méthodes actuelles avec leurs paramètres : Canny 50/150 et `approxPolyDP` à 2 % ; HSV [35–85] et [90–120], avec S et V ≥ 80 ;
- pour R2, nous avons fourni l'homographie et la position de la caméra exactes, ce qui avantage R2 ;
- pour ArUco, l'homographie est calibrée au démarrage sur 10 images, et la caméra localisée par `solvePnP`.

Chaque condition est testée sur 30 poses aléatoires. Erreurs médianes, données après correction de parallaxe ; sans cette correction, l'erreur ArUco est de 9 à 10 mm.

| Condition | Carte : contour (actuel) | Carte : coins ArUco vus | Robot : 2 pastilles HSV (actuel) | Robot : ArUco |
|---|:-:|:-:|:-:|:-:|
| Bonne lumière | 100 % | 4/4 | 100 % (0,4 mm ; 0,3°) | 100 % (1,3 mm ; 0,3°) |
| Image 2,5× plus sombre | 100 % | 4/4 | **0 %** | 100 % (1,3 mm ; 0,3°) |
| Image 4× plus sombre | **0 %** | 4/4 | **0 %** | 100 % (1,3 mm ; 0,4°) |
| Dégradé de lumière (−65 % d'un côté, fenêtre) | 100 % | 4/4 | 100 % | 100 % (1,3 mm ; 0,3°) |
| Balance des blancs chaude | 100 % | 4/4 | 100 % | 100 % (1,2 mm ; 0,4°) |
| Ombre d'une personne | 100 % | 4/4 | **37 %** | 100 % (1,2 mm ; 0,3°) |
| Robot flou de 4 px (≈ 7 mm pendant la pose) | 100 % | 4/4 | 100 % (1,1 mm) | 100 % (2,7 mm ; 0,2°) |
| Robot flou de 8 px (≈ 14 mm) | 100 % | 4/4 | 100 % (1,1 mm) | **10 %** |
| Robot flou de 12 px + JPEG qualité 60 | 100 % | 4/4 | 100 % (1,3 mm) | **0 %** |
| Main sur un coin | **0 %** | 3/4, suffisant | 100 % | 100 % (1,3 mm) |

Autres résultats :
- **Faux positifs** : aucun faux identifiant ArUco sur les 300 images.
- **Taille du marqueur robot** (image 2,5× plus sombre, JPEG 70), détections sur 30 essais :

  | Flou | 40 mm | 50 mm | 60 mm | 70 mm | 80 mm |
  |---|:-:|:-:|:-:|:-:|:-:|
  | 4 px | 24 | 29 | 30 | 30 | 26 |
  | 8 px | 0 | 0 | 1 | 17 | 26 |

  Plus le marqueur est grand, mieux il supporte le flou. À 80 mm, le marqueur simulé dépassait de l'avant arrondi du robot et ses coins étaient rognés, d'où le 26/30 à 4 px : il faut garder une marge blanche tout autour.
- **Parallaxe** : sans correction, l'erreur médiane est de 10 mm (16 mm au pire). Corrigée avec la caméra localisée par `solvePnP`, elle tombe à 1,2 mm (1,9 au pire). Avec une caméra mesurée au mètre (erreurs de 5, 5 et 10 cm), elle est de 2,6 mm (3,6 au pire). Avec une focale de $K$ fausse de ±10 %, elle reste ≤ 3 mm.
- **Temps de calcul** : 2 à 4 ms par image 720p pour la détection ArUco.
- **Répétabilité** : robot immobile, écart-type < 0,1 mm et < 0,05° en simulation. Ce chiffre est irréaliste, car le bruit simulé est trop simple : il faudra mesurer le vrai (§8.3).

**Limites** : nos images de synthèse ont des couleurs parfaites, sans reflets, sans distorsion et sans *rolling shutter*. Les chiffres absolus sont donc optimistes, et seules les tendances comptent : quelle méthode échoue, et dans quelles conditions. Quand elles sont détectées, les pastilles sont même un peu plus précises qu'ArUco en simulation (0,4 mm contre 1,3 mm). **Leur problème est la fiabilité, pas la précision.**

## 8. Conclusion : la solution recommandée

### 8.1 Le montage

1. **Fond** : un carré de 1 × 1 m blanc et **mat** (poster mat, carton plume, ou feuilles mates jointives et sans plis). Ce qui l'entoure n'a plus d'importance.
2. **Coins** : 4 marqueurs `DICT_4X4_50`.
   - Identifiants : **0** en bas à gauche (l'origine), **1** en bas à droite, **2** en haut à droite, **3** en haut à gauche.
   - Carré noir de **80 à 100 mm**, avec une marge blanche d'au moins 1 module (environ 15 mm) tout autour.
   - Tous collés **dans le même sens** : haut du marqueur vers le haut de la carte (+y), à environ 2 cm des bords.
   - Positions et taille **mesurées à la règle** : ce sont nos balises de position connue (L8).
3. **Robot** : un marqueur **n° 4** de **70 mm**.
   - Sur un carton blanc **rigide, plat et opaque** d'environ 100 × 100 mm, fixé sur le dessus du Thymio.
   - Haut du marqueur = avant du robot.
   - Idéalement, centre du marqueur au-dessus du trou du stylo, c'est-à-dire de l'axe des roues. Sinon, mesurer le décalage (environ 24 mm d'après `Sandbox/Tim/vision.py`).
   - Mesurer la hauteur h du marqueur au-dessus du sol (environ 5,5 cm).
4. **But** : un marqueur **n° 5** de 70 mm, sur un carton qu'on peut déplacer.
5. **Obstacles globaux** : des formes **plates** en carton noir mat.
   - Plates : pas de parallaxe. Noires sur blanc : contraste maximal.
   - À au moins 2 cm des marqueurs. Sinon, leur noir touche la bordure noire du marqueur, qui n'est plus un carré isolé.
6. **Obstacles locaux**, posés pendant la démo et vus par les capteurs de proximité : des objets en volume. Ils n'entrent pas dans la carte globale, qui est construite au démarrage.
7. **Caméra** :
   - fixée rigidement (trépied, étagère) ;
   - environ 1,6 m au-dessus de la carte pour une webcam 720p d'environ 70° de champ, ce qui donne environ 0,6 px/mm ; 1080p si possible ;
   - autofocus coupé ;
   - connexion USB : un natel en WiFi ajoute une latence variable.
8. **Lumière** : diffuse, sans spot qui se reflète dans le papier, sans contre-jour. Assez forte pour que l'exposition reste courte et limite le flou.
9. **Thymio** : utiliser le dongle radio plutôt que le câble USB, qui peut cacher le marqueur.

Pour générer les marqueurs à imprimer, imprimer à l'échelle 100 % puis **mesurer** le carré noir imprimé :

```python
import cv2

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
for marker_id in range(6):  # 0-3 : coins, 4 : robot, 5 : but
    image = cv2.aruco.generateImageMarker(dictionary, marker_id, 600)  # 6 modules de 100 px
    image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=255)  # marge blanche d'1 module
    cv2.imwrite(f"aruco_4x4_id{marker_id}.png", image)
```

### 8.2 Le logiciel

0. **Une seule fois (optionnel)** : calibration intrinsèque au damier, qui donne $K$ et la distorsion (§4.3). Sinon, deux options : estimer $K$ depuis l'angle de champ, ou mesurer la position de la caméra au mètre.
1. **Démarrage, environ 1 s** (robot posé, rien devant les coins) :
   1. sur 30 images, détecter les coins des marqueurs 0 à 3 et les moyenner ;
   2. `findHomography` sur les 16 points, puis test de reprojection ;
   3. `solvePnP`, qui donne la position de la caméra (N, $H_c$).
2. **Démarrage, carte des obstacles** :
   1. vue de dessus avec `warpPerspective`, à 2 mm/px ;
   2. seuillage (Otsu, ou le seuillage adaptatif actuel) ;
   3. masquer les marqueurs et le robot ;
   4. contours, puis polygones des obstacles en mm, grossis du rayon du robot, pour la navigation globale.

   Le but est le centre du marqueur n° 5.
3. **À chaque image** :
   1. détecter les marqueurs et prendre le n° 4 ;
   2. corriger ses coins de la distorsion si la caméra est calibrée ;
   3. les passer dans H pour obtenir des mm ;
   4. corriger la parallaxe ;
   5. en déduire (x, y, θ) de l'axe des roues et appeler `ekf.update`.

   Si le marqueur n'est pas vu, `ekf.predict` seul. Si la mesure tombe très loin de la prédiction (plus de 10 cm environ), c'est un kidnapping : réinitialiser le filtre sur la mesure.
4. **Surveillance** des coins, et recalibration si la caméra a bougé (§4.5).
5. **Affichage en direct** (Pièges : visualisation, paramètres en surimpression) : marqueurs détectés (`cv2.aruco.drawDetectedMarkers`), repère de la carte, chemin, pose vision et pose Kalman avec l'ellipse.

**Paramètre à ne pas oublier** : `cornerRefinementMethod = CORNER_REFINE_SUBPIX`. Par défaut, il vaut `NONE` et les coins ne sont précis qu'au pixel près. Le reste peut garder les valeurs par défaut.

**Attention aux vieux tutoriels** : depuis OpenCV 4.7, ArUco est dans le paquet principal (plus besoin d'`opencv-contrib`) et l'API a changé.
- `cv2.aruco.ArucoDetector(...).detectMarkers(...)` remplace `cv2.aruco.detectMarkers(image, dictionary)`.
- `solvePnP` remplace `estimatePoseSingleMarkers`.

Le squelette du §9 est testé avec la version installée (OpenCV 5.0.0).

### 8.3 Valider la vision et mesurer R
1. Marquer 9 croix à des positions connues de la carte : 250, 500 et 750 mm en x et en y.
2. Poser le robot avec le centre de l'axe sur chaque croix, dans 4 orientations, et enregistrer 50 mesures par position.
3. **Biais** = moyenne − vérité. Il doit rester sous 5 mm environ. Sinon, vérifier la taille et la position des marqueurs, h, la position de la caméra et la distorsion.
4. **Écarts-types** σx, σy, σθ : ils donnent `CAMERA_VARIANCES = [σx², σy², σθ²]` dans `kalman.py`. C'est mieux que la valeur « assumed » actuelle, et cela évite le piège « coefficients du filtre non mesurés ».
5. **Taux de détection** en roulant à la vitesse prévue : viser plus de 90 %. Sinon, augmenter la lumière ou réduire la vitesse.

### 8.4 Pourquoi c'est la solution la plus fiable
1. **Elle résiste à la lumière, le piège n°1 du cours.**
   - ArUco travaille en niveaux de gris, avec un seuil adaptatif local, puis lit un code binaire. Ni la couleur, ni la balance des blancs, ni les dégradés de lumière n'interviennent.
   - En simulation : robot trouvé dans 100 % des images et 4 coins sur 4 sous tous les éclairages.
   - Par comparaison, les pastilles HSV actuelles tombent à 0 % (image 2,5× plus sombre) et à 37 % (ombre). Le contour de carte tombe à 0 % en lumière très faible.
2. **Elle ne donne jamais de pose fausse.**
   - Les identifiants donnent l'origine de la carte et séparent robot, but et coins. Le marqueur n'étant pas symétrique, il donne aussi le cap sur 360°.
   - Un faux marqueur devrait réunir bordure, grille et code du dictionnaire : aucun faux identifiant sur 300 images.
   - En cas d'échec, la vision rend `None` et le Kalman continue avec l'odométrie. C'est exactement ce que demandent les tests « caméra cachée » et « kidnapping » de la démo (P13).
3. **La géométrie est juste.** L'homographie convertit en mm à partir de 16 points, et la parallaxe est corrigée. Résultat : environ 1 à 3 mm, au lieu d'environ 10 mm (jusqu'à 2 cm aux coins). Le gain de précision vient surtout de la géométrie, pas du détecteur.
4. **Elle supporte les occlusions.** H est figée au démarrage : un coin caché ensuite ne change rien. Au démarrage, 3 coins suffisent. Le contour de carte, lui, échoue dès qu'un coin est caché.
5. **Elle retrouve le robot immédiatement après un kidnapping.** La détection repart de zéro à chaque image, sans état à garder ni initialisation, contrairement au suivi meanShift.
6. **Elle est rapide** : 2 à 4 ms par image en 720p.
7. **Elle est ancrée dans le cours.** Chaque étape correspond à une notion du cours : seuillage adaptatif, contours, approximation polygonale, projection perspective, Otsu, balises de position connue, calibration et localisation de caméra, filtre bayésien.
   - Cela respecte P15 : une bibliothèque qu'on comprend et qu'on sait expliquer (§8.5).
   - Elle évite le piège « couleur ».
   - Elle reste dans l'esprit « ne pas en faire plus que nécessaire » (Pièges) : six marqueurs imprimés et un carton, aucune construction.
8. **Son point faible est connu et couvert.** Contre le flou de mouvement :
   - éclairage suffisant, donc exposition courte ;
   - vitesse modérée, environ 100 à 150 mm/s (le Thymio monte à environ 200 mm/s) ;
   - le plus grand marqueur possible ;
   - le dictionnaire 4×4, qui a les plus gros modules.

   Et le Kalman comble les images manquées : c'est son rôle.

**Pourquoi pas les autres options :**
- **Pastilles de couleur** : précises et robustes au flou, mais fragiles à la lumière (le piège du cours) et sans identifiant.
- **Contour ou Hough** : les coins n'ont pas d'identité, donc le repère peut tourner de 90°, et un seul coin caché casse tout.
- **QR codes** : faits pour stocker des données, avec des modules 3,5× plus petits.
- **AprilTag** : équivalent, mais ses modules plus petits conviennent moins à notre résolution. Reste une alternative si besoin.
- **CNN** : disproportionné.

À éviter également : mélanger couleur et ArUco (complexité sans gain), et recalculer H à chaque image (on ajoute du bruit).

### 8.5 L'expliquer à l'oral : ArUco avec les mots du cours
Voici les étapes du détecteur, d'après le tutoriel OpenCV « Detection of ArUco Markers » et Garrido-Jurado et al. (2014) :
1. **Niveaux de gris**, sans couleur : cela évite le piège de l'espace couleur.
2. **Seuillage adaptatif** (moyenne locale − C, avec plusieurs tailles de fenêtre), comme dans T §3.3 : il est insensible aux dégradés et aux ombres douces.
3. **Contours** (`findContours`), puis **approximation polygonale** (`approxPolyDP`) : on ne garde que les quadrilatères convexes assez grands. Ce sont les candidats.
4. **Suppression de la perspective** : une homographie (§4.2, la projection de V12) ramène chaque candidat à un carré « vu de face ».
5. **Seuillage d'Otsu** (T §3.3), puis comptage des pixels blancs et noirs dans chaque case de la grille : on obtient les bits.
6. **Comparaison avec le dictionnaire**, dans les 4 rotations, avec la distance de Hamming et une petite correction d'erreur. Cela donne l'identifiant et l'orientation, et rejette les faux candidats : « easy to extract and easy to match » (V8).
7. **Raffinement sous-pixel des coins** (`cornerSubPix`, par le gradient de l'image), comme pour les détecteurs de points d'intérêt (V24–25) : c'est la précision de localisation.

Il faut aussi savoir expliquer l'homographie (§4.2), la parallaxe (§4.4) et le rôle de la vision dans le Kalman : c'est la mesure absolue $z$, de covariance $R$.

### 8.6 Risques et parades

| Risque | Effet | Parade |
|---|---|---|
| Flou de mouvement : robot rapide, ou pièce sombre qui allonge la pose | robot perdu sur quelques images | déplacement pendant la pose = vitesse × temps de pose (150 mm/s × 33 ms ≈ 5 mm, sous la limite d'environ 7 mm) : lumière, vitesse ≤ 100–150 mm/s, marqueur de 70 mm ; le Kalman comble |
| Reflet sur le papier | marqueur invisible localement | papier mat, pas de plastification, lumière diffuse |
| Marqueur robot pas plat ou mal orienté | cap faux | carton rigide ; vérifier au démarrage que la flèche affichée pointe vers l'avant |
| Marqueur de coin tourné ou mal placé | homographie fausse | test de reprojection au démarrage : le squelette s'arrête au-delà de 5 mm |
| Caméra bougée pendant la démo | tout est décalé | surveillance des coins et recalibration automatique (§4.5) |
| Câble USB du Thymio sur la carte | marqueur caché | dongle radio |
| Latence du flux (natel en WiFi) | estimation du Kalman en retard | webcam USB, ou DroidCam en USB ; horodater les images |
| Distorsion de l'objectif | quelques mm d'erreur vers les bords | calibration au damier (§4.3) |
| h ou position de la caméra mal mesurées | quelques mm d'erreur | mesurer h à ±5 mm et la caméra à ±5 cm, ou utiliser `solvePnP` |
| ArUco inutilisable dans la salle (imprévu) | plus de carte | plan B : clic manuel des 4 coins (option G) |

### 8.7 Ce qui change par rapport au code actuel
- **Carte** : le plus grand quadrilatère trouvé par Canny, et `four_point_transform` refait à chaque image, sont remplacés par 4 marqueurs et une homographie vers les mm, calculée une seule fois.
- **Robot** : `detect_thymio` (pastilles vert et bleu en HSV) est remplacé par le marqueur n° 4.
- **Constantes** : `MM_PER_PIXEL`, `MARKERS_AHEAD_OF_AXLE_MM` et le retournement `FRAME_HEIGHT - v` disparaissent. H donne directement des mm dans le repère du Kalman, avec y vers le haut. Il reste deux mesures : `AXLE_OFFSET` et la hauteur h.
- **Obstacles** : `detect_walls` peut garder son seuillage, mais sur la vue de dessus en mm, avec les marqueurs et le robot masqués, et une seule fois au démarrage.
- **Kalman** : les valeurs « assumed » de `CAMERA_VARIANCES` sont remplacées par des valeurs mesurées (§8.3).

## 9. Squelette de code (testé)

Testé avec OpenCV 5.0.0 sur la scène simulée du §7 :
- avec $K$ : erreur médiane 1,2 mm (1,9 au pire) et 0,24° ;
- caméra mesurée au mètre : erreur médiane 2,6 mm (3,6 au pire) ;
- un marqueur de coin collé de travers est bien signalé par le test de reprojection.

```python
"""Carte et robot par marqueurs ArUco.
Repère carte : origine au coin bas-gauche, x vers la droite, y vers le haut, en mm ; theta anti-horaire depuis x."""
import cv2
import numpy as np

ARUCO = cv2.aruco
DICTIONARY = ARUCO.getPredefinedDictionary(ARUCO.DICT_4X4_50)
ROBOT_ID, GOAL_ID = 4, 5
MAP_SIDE = 1000.0            # [mm]
CORNER_SIDE = 80.0           # [mm] côté du carré noir imprimé, MESURÉ à la règle
INSET = 20.0                 # [mm] distance bord de la carte -> bord du marqueur
ROBOT_MARKER_HEIGHT = 55.0   # [mm] hauteur du marqueur robot au-dessus du sol, mesurée
AXLE_OFFSET = 0.0            # [mm] centre du marqueur devant l'axe des roues (0 si centré sur le trou du stylo)


def upright_square(cx, cy, side):
    """Coins d'un marqueur collé « droit » (son haut vers +y), dans l'ordre ArUco : HG, HD, BD, BG."""
    h = side / 2
    return np.array([[cx - h, cy + h], [cx + h, cy + h], [cx + h, cy - h], [cx - h, cy - h]])


_c = INSET + CORNER_SIDE / 2
CORNER_WORLD = {0: upright_square(_c, _c, CORNER_SIDE),                        # bas-gauche = origine
                1: upright_square(MAP_SIDE - _c, _c, CORNER_SIDE),             # bas-droite
                2: upright_square(MAP_SIDE - _c, MAP_SIDE - _c, CORNER_SIDE),  # haut-droite
                3: upright_square(_c, MAP_SIDE - _c, CORNER_SIDE)}             # haut-gauche


def make_detector():
    parameters = ARUCO.DetectorParameters()
    parameters.cornerRefinementMethod = ARUCO.CORNER_REFINE_SUBPIX  # par défaut : NONE (coins au pixel près)
    return ARUCO.ArucoDetector(DICTIONARY, parameters)


def detect_markers(detector, frame, K=None, dist=None):
    """{id: coins 4x2 en px}, corrigés de la distorsion si la caméra est calibrée."""
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    found = {}
    for marker_id, c in zip([] if ids is None else ids.ravel(), corners):
        c = c.reshape(-1, 1, 2).astype(np.float64)
        if K is not None:
            c = cv2.undistortPoints(c, K, dist, P=K)  # seulement 4 points : pas besoin de redresser toute l'image
        found[int(marker_id)] = c.reshape(4, 2)
    return found


def calibrate_map(detections, K=None):
    """Au démarrage, sur ~30 images : homographie image -> carte [mm] et position de la caméra [mm]."""
    corners = {i: [] for i in CORNER_WORLD}
    for found in detections:
        for i in CORNER_WORLD:
            if i in found:
                corners[i].append(found[i])
    seen = [i for i in CORNER_WORLD if corners[i]]
    if len(seen) < 3:
        raise RuntimeError(f"Coins vus : {seen}. Il en faut au moins 3.")
    image_pts = np.vstack([np.mean(corners[i], axis=0) for i in seen])  # moyenne = moins de bruit
    map_pts = np.vstack([CORNER_WORLD[i] for i in seen])
    homography, _ = cv2.findHomography(image_pts, map_pts, 0)           # moindres carrés sur 12-16 points
    reprojection = cv2.perspectiveTransform(image_pts.reshape(-1, 1, 2), homography).reshape(-1, 2)
    error_mm = np.linalg.norm(reprojection - map_pts, axis=1).max()
    if error_mm > 5:
        raise RuntimeError(f"Erreur de reprojection {error_mm:.1f} mm : un marqueur est-il tourné ou mal placé ?")
    camera_position = None
    if K is not None:  # localisation de la caméra (calibration extrinsèque), points déjà non distordus
        object_pts = np.column_stack([map_pts, np.zeros(len(map_pts))])
        _, rvec, tvec = cv2.solvePnP(object_pts, image_pts, K, None, flags=cv2.SOLVEPNP_IPPE)
        rotation, _ = cv2.Rodrigues(rvec)
        camera_position = (-rotation.T @ tvec).ravel()  # (x, y, hauteur) de la caméra dans le repère carte
    return homography, camera_position  # sans K : camera_position = np.array([x_N, y_N, H_c]) mesurée au mètre


def top_view(frame, homography, mm_per_px=2.0):
    """Vue de dessus redressée (obstacles, affichage) : 1 px = mm_per_px mm, y vers le haut.
    Si la caméra est calibrée, passer cv2.undistort(frame, K, dist) au lieu de frame."""
    n = int(MAP_SIDE / mm_per_px)
    map_to_view = np.array([[1 / mm_per_px, 0, 0], [0, -1 / mm_per_px, n], [0, 0, 1]])
    return cv2.warpPerspective(frame, map_to_view @ homography, (n, n))


def robot_pose(found, homography, camera_position):
    """Pose (x, y, theta) de l'axe des roues [mm, mm, rad] dans le repère carte, ou None si non vu."""
    if ROBOT_ID not in found:
        return None
    c = cv2.perspectiveTransform(found[ROBOT_ID].reshape(-1, 1, 2), homography).reshape(4, 2)
    forward = (c[0] + c[1]) / 2 - (c[3] + c[2]) / 2  # milieu bord bas -> milieu bord haut = avant du robot
    theta = np.arctan2(forward[1], forward[0])
    centre = c.mean(axis=0)  # point vu « à travers » le plan du sol
    # Parallaxe : le marqueur est à h au-dessus du sol, son rayon de vue touche le sol plus loin du nadir.
    nadir, camera_height = camera_position[:2], camera_position[2]
    centre = nadir + (centre - nadir) * (camera_height - ROBOT_MARKER_HEIGHT) / camera_height
    axle = centre - AXLE_OFFSET * np.array([np.cos(theta), np.sin(theta)])
    return np.array([axle[0], axle[1], theta])
```

Utilisation dans la boucle principale :

```python
K, dist = None, None  # ou les valeurs données par cv2.calibrateCamera (§4.3)
detector = make_detector()
startup = [detect_markers(detector, cap.read()[1], K, dist) for _ in range(30)]  # robot posé, coins dégagés
homography, camera_position = calibrate_map(startup, K)
if camera_position is None:  # pas de K : position de la caméra mesurée au mètre
    camera_position = np.array([520.0, 480.0, 1650.0])  # nadir x, nadir y, hauteur [mm]
while True:
    ok, frame = cap.read()
    pose = robot_pose(detect_markers(detector, frame, K, dist), homography, camera_position)
    ekf.predict(left_speed, right_speed, dt)  # odométrie, à chaque tour
    if pose is not None:
        ekf.update(pose)                       # caméra, quand le robot est vu
```

## 10. Références
- Cours MICRO-452 (F. Mondada) : slides V, L, C, P, *Project Common Pitfalls*, tutoriel OpenCV T, exercices 2, études de cas 2 et 8.
- S. Garrido-Jurado, R. Muñoz-Salinas, F. J. Madrid-Cuevas, M. J. Marín-Jiménez, « Automatic generation and detection of highly reliable fiducial markers under occlusion », *Pattern Recognition* 47(6), 2014.
- E. Olson, « AprilTag: A robust and flexible visual fiducial system », ICRA 2011.
- R. Hartley, A. Zisserman, *Multiple View Geometry in Computer Vision*, 2ᵉ éd., Cambridge University Press, 2004, chap. 4 (homographie, DLT).
- Z. Zhang, « A flexible new technique for camera calibration », *IEEE TPAMI* 22(11), 2000.
- T. Collins, A. Bartoli, « Infinitesimal Plane-Based Pose Estimation », *IJCV* 109, 2014 (`SOLVEPNP_IPPE`).
- Documentation OpenCV : tutoriels *Detection of ArUco Markers* et *Camera Calibration*.
- ISO/IEC 18004 (QR code).
