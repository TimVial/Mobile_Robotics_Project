# Filtre de Kalman pour le Thymio : l'essentiel du cours (10 min)

Tout vient du cours MICRO-452. Les numéros renvoient aux slides imprimées :
**U** = *Slides 07_08 - Uncertainties*, **L** = *Slides 06 - Localisation*, **Ex8** = solution de l'exercice 8 (Kalman).

---

## 1. Le problème : deux capteurs complémentaires

On veut la **pose** du robot $p = (x, y, \theta)$ (L18).

| | Odométrie (vitesses des roues) | Caméra (vision) |
|---|---|---|
| Ce qu'elle donne | un **déplacement** (relatif) | une **position absolue** |
| Fréquence | à chaque boucle | seulement si le robot est visible |
| Défaut | **dérive** : l'erreur s'accumule, surtout en angle (L22) | bruit, latence, peut être cachée |
| Rôle dans le filtre | **prédiction** | **correction** |

- Le Thymio n'a **pas d'encodeurs** : `motor.left.speed` est estimée par la force contre-électromotrice (back-EMF) des moteurs (solution ex. 1), à 100 Hz. C'est un signal bruité.
- La perception « tends to increase the robot's knowledge », le mouvement « tends to induce a loss of knowledge » (U17).
- **L'odométrie est l'entrée de commande $u$, pas une mesure $z$** (L19 : « encoders are not part of the measurement z ! »).

Le filtre de Bayes alterne deux étapes (U21) : **prédiction** avec le mouvement, puis **correction** avec la mesure. Le filtre de Kalman en est la version gaussienne.

## 2. L'idée du Kalman (U37)

- La croyance (*belief*) sur l'état est une **gaussienne** : une moyenne $\mu$ (la meilleure estimation) et une covariance $\Sigma$ (l'incertitude).
- Le filtre **minimise la covariance de l'erreur a posteriori**. Il fusionne les capteurs « intelligemment » en pesant chacun selon sa variance.
- **Pourquoi Kalman** plutôt qu'un filtre à particules ou à grille (U54, études de cas 8) : il est *compact* et rapide. Ses limites : une seule hypothèse (unimodal), il faut une bonne initialisation, il peut diverger. Ici ces limites ne gênent pas, car la caméra mesure directement la pose : il n'y a jamais d'ambiguïté entre plusieurs positions possibles.

## 3. Le cas scalaire : toute l'intuition (U38-41)

Modèle : $x_{t+1} = x_t + d + w_t$ avec un bruit de mouvement de variance $q$, et $y_t = x_t + v_t$ avec un bruit de mesure de variance $r$.

| Étape | Formule | Sens |
|---|---|---|
| Prédiction | $\bar\mu_t = \mu_{t-1} + d$ | on avance avec le mouvement |
| | $\bar\Sigma_t = \Sigma_{t-1} + q$ | l'incertitude **augmente** |
| Innovation | $i_t = y_t - \bar\mu_t$ | écart entre la mesure et la prédiction |
| Gain | $K_t = \dfrac{\bar\Sigma_t}{\bar\Sigma_t + r} \in [0, 1]$ | la part de l'innovation que l'on accepte |
| Correction | $\mu_t = \bar\mu_t + K_t\, i_t$ | |
| | $\Sigma_t = (1 - K_t)\,\bar\Sigma_t$ | l'incertitude **diminue** |

Lecture du gain :
- $r \to 0$ (capteur parfait) : $K \to 1$, on recopie la mesure.
- $r \to \infty$ (capteur inutile) : $K \to 0$, on garde la prédiction.
- $q$ grand (mouvement mal connu) : $\bar\Sigma$ grand, donc $K \to 1$.

## 4. Le cas matriciel (U42-46)

$x_{t+1} = A x_t + B u_t + w_t$ avec $w \sim \mathcal N(0, Q)$, et $y_t = C x_t + v_t$ avec $v \sim \mathcal N(0, R)$.

$$\bar\mu_t = A\mu_{t-1} + Bu_t \qquad \bar\Sigma_t = A\Sigma_{t-1}A^T + Q$$
$$i_t = y_t - C\bar\mu_t \qquad S_t = C\bar\Sigma_t C^T + R \qquad K_t = \bar\Sigma_t C^T S_t^{-1}$$
$$\mu_t = \bar\mu_t + K_t i_t \qquad \Sigma_t = (I - K_t C)\,\bar\Sigma_t$$

- $\bar\Sigma = A\Sigma A^T + Q$ : on propage l'incertitude existante à travers le mouvement, puis on ajoute celle du mouvement lui-même.
- $S$ est l'incertitude de l'innovation : celle de la prédiction plus le bruit du capteur.
- Dans le code, $C$ s'appelle **H**, comme dans l'ex. 8 et dans les slides EKF.

## 5. Pourquoi un filtre de Kalman **étendu** (EKF, U47-48)

Le mouvement du robot contient $\cos\theta$ et $\sin\theta$, donc il **n'est pas linéaire** : on ne peut pas l'écrire $A\mu + Bu$.
L'EKF garde les mêmes équations, mais :
- la moyenne est prédite avec le vrai modèle non linéaire : $\bar\mu_t = g(u_t, \mu_{t-1})$ ;
- la covariance est propagée avec la **jacobienne** $G_t = \partial g / \partial \mu$, c'est-à-dire avec $g$ linéarisé autour de l'estimation courante.

> **Piège de notation.** Les slides EKF (U47-48, notation de Thrun) appellent $R_t$ le bruit du **mouvement** et $Q_t$ le bruit de **mesure**. C'est l'inverse des slides U42-46 et de l'ex. 8. Dans notre code : **Q = mouvement, R = caméra**.

## 6. Notre filtre, équation par équation (`kalman.py`)

**État** : $\mu = (x, y, \theta)$ en mm, mm et rad. **Entrée** : les vitesses mesurées des roues $v_l, v_r$ en mm/s, pendant $dt$.

**Prédiction** (odométrie de L18, avec $b$ = écart entre les roues) :

$$\Delta s = \frac{v_r + v_l}{2}\,dt \qquad \Delta\theta = \frac{v_r - v_l}{b}\,dt$$
$$\bar\mu = \mu + \begin{bmatrix}\Delta s\cos(\theta + \Delta\theta/2)\\ \Delta s\sin(\theta + \Delta\theta/2)\\ \Delta\theta\end{bmatrix}
\qquad
G = \begin{bmatrix}1 & 0 & -\Delta s\sin(\theta + \Delta\theta/2)\\ 0 & 1 & \Delta s\cos(\theta + \Delta\theta/2)\\ 0 & 0 & 1\end{bmatrix}$$
$$Q = \mathrm{diag}(\sigma_v^2,\ \sigma_v^2,\ \sigma_\omega^2)\,dt^2 \qquad \bar\Sigma = G\Sigma G^T + Q$$

**Correction** (la caméra mesure directement la pose, $z = (x_{cam}, y_{cam}, \theta_{cam})$) :
- $H = I_3$, donc la mesure est **linéaire** et il n'y a pas de jacobienne à calculer.
- $i = z - H\bar\mu$, avec l'angle de $i$ ramené dans $[-\pi, \pi[$ : sinon 179° contre −179° donnerait 358° d'erreur au lieu de 2°.
- $S = H\bar\Sigma H^T + R$, $K = \bar\Sigma H^T S^{-1}$, $\mu = \bar\mu + K i$, $\Sigma = (I - KH)\bar\Sigma$.
- Avec $H = I$, on obtient $K = \bar\Sigma(\bar\Sigma + R)^{-1}$ : c'est **exactement le cas scalaire** du §3, en matrices.

**Caméra cachée** : pas de correction, donc $\mu = \bar\mu$ et $\Sigma = \bar\Sigma$. L'ellipse d'incertitude grandit, surtout **perpendiculairement au mouvement** : l'erreur d'angle se transforme en erreur latérale (L22 : « the error is bigger in angle than straight »).

**Correspondance code et symboles** :

| Code | Symbole | Code | Symbole |
|---|---|---|---|
| `pose` | $\mu$ | `measurement_matrix` | $H$ |
| `covariance` | $\Sigma$ | `camera_noise` | $R$ |
| `motion_jacobian` | $G$ | `innovation` | $i$ |
| `motion_noise` | $Q$ | `innovation_covariance` | $S$ |
| `wheel_distance` | $b$ | `kalman_gain` | $K$ |

**Côté vision** (`vision.py`), le point suivi est le **centre de l'essieu**. Le trou du stylo, qui est le centre de rotation, se trouve à 22 % de la taille du robot derrière le centre de sa silhouette (fiche Thymio du cours et vidéo vue de dessus). Sans cette correction, chaque rotation sur place ferait « bouger » la position vue par la caméra alors que l'odométrie ne prédit aucun déplacement.

## 7. Choix des constantes : le plus important pour les questions

Le cours (*Project Common Pitfalls*) reproche de **ne pas définir les coefficients expérimentalement**, de prendre les valeurs par défaut d'une librairie ou de faire des estimations théoriques sans lien avec la réalité. Tout se mesure donc dans la section 4 du notebook, avec la méthode de l'ex. 8 : des variances calculées sur des données enregistrées.

| Constante | Défaut | Comment on la mesure | Source |
|---|---|---|---|
| $b$ `WHEEL_DISTANCE_MM` | 95 mm | dimension du Thymio | fiche Thymio |
| `SPEED_TO_MM_S` | 0.43 | distance vue par la caméra ÷ ∫ vitesse mesurée, sur une ligne droite de 4 s | Ex8 (0.43) ; fiche : 500 ≈ 20 cm/s |
| $\sigma_v^2$, $\sigma_\omega^2$ → $Q$ | 5.4 mm²/s², 1.6e-3 rad²/s² | variance des vitesses mesurées à consigne constante, après la phase d'accélération | méthode Ex8 |
| $R$ `CAMERA_VARIANCES` | 1 mm², 1 mm², (1°)² | variance de la pose caméra, robot immobile pendant 3 s | Ex8 : « estimated from measurement data » |
| $\Sigma_0$ | $R$ | le filtre démarre sur la 1re mesure caméra | Ex8 : $\Sigma$ converge vite quel que soit $\Sigma_0$ |
| `MM_PER_PIXEL` | | 111 mm ÷ taille du robot en pixels | taille du Thymio, sans calibration |

**Pourquoi $Q \propto dt^2$** : une erreur de vitesse $\delta v$ pendant $dt$ donne une erreur de position $\delta v\,dt$, dont la variance est $\sigma_v^2\,dt^2$. $Q$ est diagonale par simplicité, comme dans l'ex. 8.

**Pourquoi toute la variance va dans Q** : dans l'ex. 8, la vitesse était à la fois un état et une mesure, et la solution partage la variance moitié dans $q$, moitié dans $r$. Ici la vitesse n'est qu'une entrée $u$, donc toute son incertitude va dans $Q$.

> **Erreur d'unité dans la solution de l'ex. 8** (cellule « Speed variance ») : la variance est calculée sur `avg_speed / facteur` au lieu de `avg_speed × facteur`. Elle vaut 151 mm²/s² au lieu de **5.4 mm²/s²**, recalculé sur les données du cours. Il faut toujours convertir en mm/s **avant** de calculer la variance.

**Validation de Q (indispensable)** : on cache la caméra (touche `c`) quelques secondes. Au retour, la vraie position doit être dans l'ellipse à 95 %, c'est-à-dire $d^2 = e^T\Sigma^{-1}e < 5.99$. La dernière cellule du notebook fait ce calcul.
La variance de vitesse ne contient **que le bruit**. Elle ignore les erreurs systématiques : facteur de conversion approximatif, roues de diamètres légèrement différents, glissement, vitesses rafraîchies moins souvent que la boucle. $Q$ est alors trop petit et le filtre trop sûr de lui.
**Règle** : multiplier $\sigma_v^2$ et $\sigma_\omega^2$ par environ $d^2/2$, puis refaire le test. En simulation, avec 1 % d'écart entre les roues, on obtient $d^2 = 63$ ; avec $Q \times 31$, on obtient $d^2 = 2.4$.

**Effet des réglages** (réponses de l'ex. 8, complétées) :

| Réglage | Effet |
|---|---|
| $Q$ grand | on fait plus confiance à la caméra ($K \to 1$) : réagit vite (kidnapping), lisse moins ; l'ellipse grandit vite sans caméra |
| $Q$ petit | on fait confiance à l'odométrie : estimation lisse, lente à accepter un kidnapping, trop confiante si $Q$ est sous-estimé |
| $R = 0$ | estimation = mesure ($K = I$, $\Sigma = 0$ après correction) |
| $R$ très grand | équivaut à $H = 0$ : la mesure n'apporte rien, l'estimation reste la prédiction |
| $\Sigma_0$ grand | « je ne sais pas où je suis » : la 1re mesure est recopiée, puis $\Sigma$ converge |

**Limites à connaître** :
- la latence du flux du natel (0.1 à 0.3 s) n'est pas modélisée ;
- $R$ mesurée robot immobile sous-estime le bruit en mouvement (flou) ;
- croyance gaussienne unimodale ;
- linéarisation de $g$, valable car $dt$ est petit ;
- $Q$ diagonale ;
- caméra non calibrée : la perspective et la distorsion faussent la position d'environ 1 cm au bord de l'image.

## 8. Questions probables et réponses courtes

- **Pourquoi un EKF ?** Le modèle de mouvement $g$ est non linéaire ($\cos\theta$, $\sin\theta$). La mesure, elle, est linéaire ($H = I$).
- **Pourquoi l'odométrie en prédiction et pas en mesure ?** C'est une entrée de commande (L19). Elle est fréquente mais dérive. La caméra est absolue et corrige la dérive.
- **Que se passe-t-il quand on cache la caméra ?** Il n'y a plus que des prédictions : $\Sigma$ augmente à chaque pas et l'ellipse grandit. Au retour de la caméra, $K$ est grand, l'estimation saute vers la mesure et l'ellipse se referme.
- **Kidnapping ?** L'odométrie ne voit rien (les roues tournent dans le vide ou pas du tout). L'innovation devient énorme et l'estimation rejoint la caméra en quelques images, d'autant plus vite que $K$ est grand. Sans caméra, le kidnapping est indétectable.
- **Pourquoi ramener l'angle dans $[-\pi, \pi[$ ?** Pour que l'innovation et l'état restent continus autour de ±180°.
- **Comment avez-vous choisi Q et R ?** Par des mesures (section 4) : variance de la pose caméra robot immobile pour $R$, variance des vitesses de roues à consigne constante pour $Q$. Puis validation en cachant la caméra (test $d^2 < 5.99$) et ajustement de $Q$.
- **À quoi sert $G$ ?** À propager l'incertitude de l'angle sur la position : une erreur de cap fait dériver le robot latéralement.
- **Le filtre peut-il diverger ?** C'est un risque de l'EKF (U54) en cas de mauvaise initialisation ou de $Q$/$R$ faux. Ici la caméra observe tout l'état, donc chaque correction ramène l'estimation.

## 9. Une boucle complète (résumé du notebook)

```
à chaque image :
    vitesses = vitesses mesurées des roues (Thymio) × SPEED_TO_MM_S
    filtre.predict(v_gauche, v_droite, dt)     # odométrie : μ̄, Σ̄
    pose = vision.detect_robot(image)          # None si invisible
    si pose : filtre.update(pose en mm)        # correction : μ, Σ, K
```
