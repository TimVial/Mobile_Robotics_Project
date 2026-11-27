# Banc de test du filtre de Kalman : Thymio + natel

Un natel filme le Thymio d'en haut. Le filtre de Kalman fusionne l'odométrie (vitesses des roues) et la pose vue par la caméra.

| Fichier | Rôle |
|---|---|
| `kalman.py` | filtre de Kalman étendu : `predict` (odométrie) et `update` (caméra) |
| `vision.py` | détection du robot dans l'image : position et angle, sans calibration |
| `kalman_test.ipynb` | notebook : connexion, mesure des constantes, filtre en direct, graphiques |
| `THEORIE_KALMAN.md` | la théorie du cours et le choix des constantes, à lire avant |

---

## 1. Installation (une seule fois)

1. **Paquets Python**. Sur ton PC, seul `tdmclient` manque. Dans un terminal :
   ```
   pip install tdmclient opencv-python numpy matplotlib notebook
   ```
2. **Thymio Device Manager (TDM)**, le petit programme qui relaie le robot vers Python. Sous Windows, installe **Thymio Suite** ([thymio.org](https://www.thymio.org/download-thymio-suite-redirect/)), qui contient le TDM, et **laisse sa fenêtre ouverte** pendant que tu travailles.
   Le `TDMLauncher.exe` du cours ne suffit pas sous Windows : ce n'est qu'un lanceur, et il affiche une icône barrée s'il ne trouve pas le TDM à côté de lui.
   Si tu ouvres Aseba Studio, **déverrouille le robot** (petit cadenas dans l'onglet), sinon Python ne peut pas le piloter.

## 2. Le natel comme caméra

1. Installe l'app **DroidCam** : sur Android depuis le Play Store, sur iPhone depuis l'App Store (*DroidCam Webcam & OBS Camera*).
2. Mets le natel et le PC **sur le même WiFi**, puis ouvre l'app. Elle affiche une adresse **WiFi IP** (ex. `192.168.1.12`) et un **port** (`4747`).
3. Teste dans le navigateur du PC : `http://192.168.1.12:4747/video`. Tu dois voir l'image. **Ferme ensuite cet onglet**, car le flux n'accepte qu'un client à la fois.
4. Dans le notebook, section 1 : `CAMERA_SOURCE = "http://192.168.1.12:4747/video"` (avec ton IP).

Si le PC ne voit pas le flux :
- **WiFi de l'EPFL** : les appareils y sont souvent isolés les uns des autres. Active le **partage de connexion** du natel, connecte le PC dessus et reprends l'IP affichée par l'app.
- **Plan B**, si l'URL ne marche pas (fréquent sur iPhone) : installe le client Windows DroidCam (droidcam.app). Il crée une webcam virtuelle : mets `CAMERA_SOURCE = 1`, ou `0` ou `2` selon le PC.

## 3. Installer la scène

- **Natel fixe au-dessus du robot** : trépied, étagère, ou règle posée entre deux chaises. Caméra arrière, écran **parallèle au sol**, format paysage, **au moins 1 m de haut**. La zone de test doit être au centre de l'image.
- **Sol sombre et mat** (tissu ou carton noir, table foncée) et **aucun autre objet blanc ou clair** dans le champ. Éclairage uniforme, sans reflet.
- Natel **branché au chargeur** et app DroidCam ouverte au premier plan.
- Le dessus blanc du robot doit être entièrement visible. Un robot qui touche le bord de l'image est ignoré volontairement, pour éviter une mesure fausse.

## 4. Le Thymio

- Allume-le et branche-le en **USB** (ou utilise le dongle RF). Le TDM doit tourner.
- Aucun autre programme ne doit garder le robot : ni Aseba Studio, ni un autre notebook.
- **Rien à téléverser à la main** : la section 2 du notebook charge elle-même sur le robot un mini-programme Aseba qui applique les vitesses envoyées par le PC.

## 5. Lancer le notebook

Ouvre un terminal **dans le dossier de ce README** (`Sandbox/Tim`) puis tape `jupyter notebook kalman_test.ipynb`. Tu peux aussi l'ouvrir dans VS Code.
Exécute les cellules **dans l'ordre**. Les fenêtres d'images s'ouvrent à part, pas dans le notebook : **clique sur la fenêtre avant d'appuyer sur une touche**.

| Section | Ce qu'il faut faire / voir |
|---|---|
| 1. Settings | Mettre ton `CAMERA_SOURCE`. |
| 2. Connect | Le message « Connected to … » doit s'afficher. |
| 3. Camera check | Règle le **curseur `white threshold`** de la fenêtre *mask* jusqu'à ce que **seul le robot soit blanc**. Trop bas, le fond devient blanc ; trop haut, le robot se troue puis disparaît. Prends le milieu de la plage qui marche (souvent entre 50 et 150 sur fond noir ; 90 par défaut). La flèche verte part de l'axe des roues et pointe vers l'**avant arrondi**. **Contrôle du miroir** : tourne le robot à la main vers sa gauche (sens anti-horaire vu d'en haut), l'angle affiché doit **augmenter**. Sinon l'image est en miroir : désactive le miroir dans l'app. Laisse ensuite le robot immobile et appuie sur `q` : l'échelle `MM_PER_PIXEL` est mesurée sur 30 images. **Copie la valeur `WHITE_THRESHOLD` affichée dans la section 1** pour la garder. |
| 4. Mesures | Robot au centre de l'image, environ 25 cm libres devant lui. Il reste immobile 3 s puis roule tout droit 4 s. **Copie les constantes affichées dans la section 1** pour les garder. |
| 5. Filtre | Le robot tourne en rond. Vert = caméra, rouge = odométrie seule, bleu = Kalman avec son ellipse à 95 %. Touche `c` : caméra on/off. Touche `q` : arrêt. |
| 6. Résultats | Trajectoires, écarts-types et **validation de Q** (voir `THEORIE_KALMAN.md`, §7). La validation exige d'avoir coupé la caméra avec `c` : une main devant l'objectif cache aussi la vérité terrain. |
| 7. Disconnect | Libère la caméra et le robot. |

## 6. Les tests à faire (et à montrer)

1. **Caméra active** : le bleu colle au vert, l'ellipse est minuscule et le rouge dérive lentement.
2. **Caméra cachée** (touche `c`, ou une main devant l'objectif) : le bleu continue avec l'odométrie et l'ellipse grandit. La flèche verte, qui reste affichée avec la touche `c`, doit rester dans l'ellipse. Sinon, augmenter Q.
3. **Retour de la caméra** : le bleu rejoint le vert et l'ellipse se referme. Le gain K affiché saute vers 1.
4. **Kidnapping** : soulève le robot et pose-le ailleurs. Le rouge reste faux et le bleu rejoint le vert en quelques images.

Repère utilisé : origine en **bas à gauche** de l'image, x vers la droite, y vers le haut, en mm. θ est compté dans le sens trigonométrique depuis l'axe x : 0 = robot tourné vers la droite.

## 7. Dépannage

| Symptôme | Cause probable | Solution |
|---|---|---|
| `Cannot open the camera` | mauvaise IP, autre réseau, onglet du navigateur encore ouvert | revoir le §2, fermer le navigateur, rouvrir l'app |
| L'image prend de plus en plus de retard | flux trop lourd | baisser la résolution ou les FPS dans l'app |
| Des éléments du fond sont blancs dans *mask* | seuil trop bas | monter le curseur. Un reste de fond blanc est toléré tant qu'il ne touche pas le robot : seule une forme de robot (presque carrée, bien remplie) est retenue |
| Le robot a des trous ou disparaît du *mask* | seuil trop haut | baisser le curseur |
| Aucun seuil ne sépare le robot du fond | fond presque aussi clair que le robot | sol plus sombre et mat, enlever les objets clairs et les reflets, éviter le contre-jour |
| Pas de flèche verte | robot pas seul en blanc, ou touchant le bord de l'image | régler le seuil (lignes ci-dessus), éloigner le robot du bord |
| Flèche à l'envers ou de travers | dessus du robot partiellement caché, ou natel pas parallèle au sol | dégager le robot, redresser le natel, le monter plus haut |
| Section 2 bloquée sur `wait_for_node` | TDM arrêté, robot éteint ou câble | démarrer le TDM, allumer le robot |
| `Node lock error` | un autre programme garde le robot | fermer Aseba Studio et les autres notebooks, redémarrer le kernel, débrancher puis rebrancher |
| L'estimation saute ou dérive | constantes non mesurées | refaire la section 4, puis la validation de Q (section 6) |

## 8. Pour le projet final

- `kalman.py` ne dépend ni de la caméra ni du Thymio. Dans la boucle du projet, il suffit d'appeler `predict` à chaque tour avec les vitesses mesurées en mm/s, et `update` quand la vision donne une pose (x, y, θ) en mm dans le même repère.
- `vision.py` sert seulement à tester le filtre. La vision de l'équipe pourra le remplacer : il suffira de remesurer sa variance $R$ (section 4).
