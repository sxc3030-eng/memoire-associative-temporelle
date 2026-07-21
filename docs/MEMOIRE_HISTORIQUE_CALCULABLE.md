# Mémoire historique calculable

## Objectif expérimental

Le laboratoire historique met la mémoire associative à l'épreuve avec des
faits datés, des noms ambigus, des changements d'état, des doublons, des
contradictions et un ordre de réception différent de l'ordre historique.
Chaque exécution utilise ses propres bases SQLite temporaires. Elle ne lit ni
ne modifie la mémoire principale de l'agent.

Deux horloges restent séparées :

- `valid_from` et `valid_to` décrivent la période pendant laquelle un fait est
  présenté comme valable ;
- l'ordre d'injection décrit le moment auquel la mémoire reçoit ce fait.

Le moteur de rappel v1 classe encore principalement les épisodes à partir des
mots et du temps d'ingestion. Le rapport du laboratoire ne prétend donc pas
qu'il sait déjà résoudre toutes les questions « vrai à la date T » : il mesure
précisément cet écart.

## Un fait source n'est pas un résultat calculé

Un fait historique source contient au minimum un identifiant, un sujet, une
relation, une valeur, une date de début et une source. Les mesures, coordonnées,
dates de fin, corrections et relations de remplacement sont facultatives.

Les résultats calculés sont conservés comme des dérivations, avec :

- la formule déterministe utilisée ;
- les valeurs d'entrée et leurs unités ;
- les identifiants des faits dont elles dépendent ;
- la version du moteur de calcul ;
- le caractère exact ou approché du résultat.

Une dérivation ne devient jamais une nouvelle preuve indépendante de ses
propres entrées. Dans la mémoire de test, elle utilise la provenance `inferred`
et ne renforce donc pas les continuations factuelles.

## Données calculables prises en charge

Le registre fini annonce exactement **11 familles**. Une famille ne produit un
résultat que lorsque toutes ses entrées sont présentes et compatibles :

1. `duration_years` — durée en années civiles ;
2. `duration_months` — durée en mois civils ;
3. `duration_days` — durée exacte en jours ;
4. `temporal_midpoint` — milieu d'un intervalle ;
5. `interval_from_previous` — temps depuis le fait précédent ;
6. `age_at_start` — âge à la date de début du fait ;
7. `normalized_measurement` — conversion vers l'unité canonique ;
8. `absolute_change` — variation absolue ;
9. `percent_change` — variation en pourcentage ;
10. `annual_rate` — taux de variation annuel ;
11. `distance_from_previous` — distance géographique entre deux coordonnées.

Les longueurs, masses, durées et températures utilisent un registre d'unités
explicite. Une mesure dont l'unité est inconnue est conservée telle quelle dans
le fait normalisé, marquée `calculable: false` et accompagnée de
`skip_reason: unsupported_or_unsourced_unit`. Elle reste donc rappelable comme
donnée opaque, mais aucune dérivation numérique n'est fabriquée à partir
d'elle. Deux dimensions incompatibles ne sont pas comparées. Les monnaies ne
sont pas converties sans table de taux datée et sourcée.

Le calendrier civil n'a pas d'année zéro : `-1` représente 1 avant notre ère
et `1` représente l'an 1 de notre ère. Le calcul interne tient compte de ce
passage pour éviter d'ajouter une année inexistante.

## Scénario de torture

Un profil reproductible génère une chronique fictive couvrant l'Antiquité
jusqu'à 2026. Le corpus fictif permet de connaître toutes les bonnes réponses
sans présenter une interprétation historique contestée comme une vérité
absolue. Il introduit volontairement :

- des entités aux noms proches dans plusieurs régions ;
- des événements reçus hors ordre ;
- des répétitions portant la même clé d'idempotence ;
- des affirmations incompatibles conservées comme contradictions ;
- des mesures et coordonnées permettant des calculs dérivés.

La vérité de référence est construite directement depuis le scénario. Elle ne
réutilise ni le classement, ni les tables, ni le découpage lexical du moteur de
mémoire.

Le rapport sépare deux niveaux de mesure :

- le score sémantique top 1/top 5 utilise des questions naturelles contrôlées
  sur les dates, l'état le plus récent, les contextes et les contradictions ;
- les marqueurs exacts des faits et des dérivations servent uniquement de
  diagnostic de plomberie. Ils vérifient le routage vers le bon épisode et
  sont exclus du score sémantique.

Une contradiction exige la présence de ses deux épisodes dans le top 5 et
n'entre pas dans le dénominateur top 1. Surtout, le rappel n'est exécuté et
scoré que si le pipeline est entièrement vidé, sans échec et avec exactement le
nombre attendu de travaux terminés. Sinon le run est `incomplete` et le rappel
est `not_scored` avec des pourcentages `null`.

## Passage aux archives historiques réelles

Le format local est conçu pour recevoir ensuite des sous-ensembles historiques
sourcés. Une source possible est un export JSON hors ligne de Wikidata : le
format complet conserve les quantités, dates, unités, qualificatifs et
références. Le dépôt ne télécharge pas automatiquement ce corpus et le moteur
n'appelle aucune API externe. Un gros export devra être filtré et validé avant
injection ; les affirmations contestées devront rester multiples et conserver
leurs références.

Documentation officielle :

- <https://www.wikidata.org/wiki/Wikidata:Database_download/en>
- <https://www.mediawiki.org/wiki/Wikibase/DataModel>

## Exécution locale

```bash
python scripts/benchmark_history.py --count 25 --seed 20260721
```

Le rapport JSON indique toujours le nombre de faits sources, de dérivations,
de doublons, de contradictions, l'état du pipeline, les latences, le débit, les
tailles SQLite et les limites connues. Les scores sémantiques et le diagnostic
de plomberie y sont publiés séparément. Augmenter progressivement `--count`
est volontaire : le moteur actuel reconstruit encore certains agrégats globaux
après chaque observation, ce que le laboratoire doit justement rendre visible.
