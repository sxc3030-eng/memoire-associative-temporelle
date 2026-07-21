# Curriculum scientifique et biographique v1

## Objectif

Ce petit corpus réel met à l'épreuve une mémoire partagée avec des rôles, des
dates, des lieux, des contributions multiples et des conséquences documentées.
Il ne cherche pas à désigner artificiellement un inventeur unique.

Les cinq dossiers initiaux sont :

1. Darwin et Wallace, dont les textes sur la sélection naturelle ont été lus à
   la Linnean Society le 1er juillet 1858, alors qu'aucun des deux n'était
   présent ;
2. Marie et Pierre Curie, avec des rôles communs dans les découvertes du
   polonium et du radium et un travail d'isolement distinct ;
3. la pénicilline, en séparant l'observation de Fleming en 1928 de son effet
   thérapeutique et des contributions de Chain et Florey ;
4. la structure de l'ADN, en séparant les données expérimentales de Franklin et
   Gosling, les contributions de Wilkins et la modélisation de Watson et Crick ;
5. le World Wide Web, inventé par Tim Berners-Lee au CERN en 1989 puis rendu
   disponible sur une base libre de redevances par le CERN le 30 avril 1993.

## Sources institutionnelles

- [Linnean Society — présentation Darwin–Wallace de 1858](https://www.linnean.org/news/2018/07/01/1st-july-2018-160th-anniversary-of-the-presentation-of-on-the-tendency-of-species-to-form-varieties)
- [Nobel — conférence de Marie Curie sur le radium](https://www.nobelprize.org/prizes/chemistry/1911/marie-curie/lecture/)
- [Imperial College Healthcare NHS Trust — musée Fleming](https://www.imperial.nhs.uk/about-us/what-we-do/fleming-museum)
- [Nobel — prix de médecine 1945 à Fleming, Chain et Florey](https://www.nobelprize.org/prizes/medicine/1945/summary/)
- [King's College London — histoire de la photographie 51](https://www.kcl.ac.uk/the-story-behind-photograph-51)
- [Université de Cambridge — contributions à la double hélice](https://www.cam.ac.uk/stories/DNA-structure-discovery-cambridge-70th-anniversary)
- [CERN — courte histoire du Web](https://home.cern/science/computing/the-birth-of-the-web/short-history-web/)

## Contrat des affirmations

Chaque affirmation contient :

- un identifiant stable ;
- le sujet, la relation et l'objet sous forme d'identifiants d'entité ;
- le rôle exact (`experimental_data`, `observation`, `model_building`, etc.) ;
- une date avec sa précision réelle ;
- un lieu identifié lorsque la source le permet ;
- les sources qui soutiennent directement l'affirmation ;
- un statut d'attribution ;
- les remplacements ou rétractations éventuels.

`presented_at` ne signifie pas `present_at_place`. `patent_holder` ne signifie
pas `inventor`. Une succession chronologique ne signifie pas `caused`. Une
coprésence compatible ne prouve jamais une rencontre.

## Questions tenues à l'écart

Les questions d'évaluation se trouvent dans `evaluation_questions`, séparées
des affirmations importées. Les fragments attendus et interdits servent au
correcteur, pas au prompt du modèle.

Le serveur utilise l'importeur scientifique spécialisé pour construire une
base persistante séparée. Il ne faut jamais envoyer le fichier complet à
l'import JSON générique, car sa partition d'évaluation contient volontairement
la grille de correction. L'importeur spécialisé projette uniquement les
sources, les entités et les affirmations avant toute écriture SQLite.

Le premier lot vérifie notamment :

- l'absence physique de Darwin et Wallace lors de la lecture de 1858 ;
- la collaboration de Marie et Pierre Curie sans effacer leurs tâches
  distinctes ;
- la différence entre découverte et développement thérapeutique de la
  pénicilline ;
- la distinction entre production de données et construction d'un modèle pour
  l'ADN ;
- la différence entre invention du Web et décision ultérieure du CERN de
  publier le code sans redevances.

## Limites

Le corpus est volontairement petit. Il vérifie le contrat et la fidélité aux
preuves, pas une connaissance générale de l'histoire des sciences. Les textes
de réponse peuvent varier entre modèles ; les fragments attendus doivent donc
rester factuels et ne jamais imposer une formulation stylistique unique.
