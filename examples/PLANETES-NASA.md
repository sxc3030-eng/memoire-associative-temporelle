# Jeu de mémoire : les huit planètes

Le fichier [`planetes-nasa.json`](planetes-nasa.json) est conçu pour tester le rappel du prototype. Chaque planète tient dans une seule chaîne afin qu'une question portant sur son nom retrouve toute sa fiche dans un même souvenir.

Les valeurs sont arrondies comme dans les pages NASA consultées :

- [Planet Sizes and Locations in Our Solar System](https://science.nasa.gov/solar-system/planets/planet-sizes-and-locations-in-our-solar-system/) — ordre, rangs par taille, diamètres équatoriaux et distances moyennes au Soleil;
- [About the Planets](https://science.nasa.gov/solar-system/planets/) — huit planètes et classification en planètes telluriques, géantes gazeuses et géantes de glace.

Questions à essayer après l'import :

```text
Que sais-tu de Mars ?
Quelle est la plus grande planète ?
Quelles planètes sont des géantes de glace ?
Quel est le diamètre de Saturne ?
Quelle planète est la plus éloignée du Soleil ?
D'où viennent les données sur les planètes ?
```

Le prototype effectue encore un rappel lexical : il retrouve et classe des souvenirs, mais ne calcule pas une réponse nouvelle comme le ferait un modèle de langage.
