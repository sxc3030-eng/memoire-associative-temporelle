# Benchmark du laboratoire historique v0.5

## Ce qui est mesuré

Le benchmark génère une chronique fictive reproductible, construit sa vérité
de référence avec un oracle indépendant, calcule les dérivations disponibles,
puis injecte les faits sources comme `observed` et les calculs comme `inferred`
dans un `MemoryPipeline` temporaire.

Le catalogue annonce exactement **11 familles de calcul** : durées civiles en
années, mois et jours, milieu temporel, intervalle depuis le fait précédent,
âge au début du fait, normalisation d'une mesure, variation absolue, variation
en pourcentage, taux annuel et distance géographique.

Il vérifie :

- rappel naturel par date, état le plus récent, contexte et contradiction ;
- routage exact par marqueur propre au fait ou à une valeur dérivée ;
- séparation d'homonymes par contexte ;
- présence des deux affirmations d'une contradiction ;
- déduplication exacte des soumissions rejouées ;
- nombre attendu d'événements et type de provenance ;
- suppression des bases temporaires après le test.

Deux groupes de résultats sont volontairement séparés :

- le **score sémantique** top 1/top 5 porte sur les formulations naturelles
  contrôlées, sans identifiant ni marqueur synthétique dans la requête ;
- le **diagnostic de plomberie** porte sur les marqueurs exacts des faits et
  des calculs dérivés. Il vérifie le câblage index → épisode, mais ne contribue
  jamais au score sémantique publié.

Pour une contradiction, le top 5 exige que les deux épisodes indépendants
soient présents. Cette question n'est pas admissible au top 1, qui ne peut
matériellement contenir deux réponses.

Le score n'est calculé que si la file est entièrement vidée, qu'aucun ticket
n'a échoué et que le nombre de travaux terminés est exactement celui attendu.
Sinon le rapport porte le statut `incomplete`, le rappel porte le statut
`not_scored`, aucune question n'est exécutée et les pourcentages restent
`null`. Un pipeline partiel ne peut donc pas produire un score trompeur.

Ce benchmark ne mesure pas encore une compréhension libre de l'histoire et ne
prouve pas un raisonnement bitemporel natif. Les requêtes sont construites par
l'oracle avec des indices dont la bonne réponse est connue.

## Conditions locales

```text
Système : Windows 11, build 26200
Python : 3.12.13
Processeur rapporté : Intel64 Family 6 Model 198 Stepping 2
Date : 2026-07-21
```

Les résultats dépendent du disque, du processeur, des processus concurrents et
de l'état du système. Ils ne doivent pas être extrapolés directement à une
autre machine.

## Exécution à 25 faits sources

Commande :

```bash
python scripts/benchmark_history.py --count 25 --seed 20260721
```

Résultat :

| Mesure | Valeur |
|---|---:|
| Faits sources de base | 25 |
| Contradictions ajoutées | 2 |
| Valeurs calculées | 140 |
| Soumissions totales | 170 |
| Événements uniques | 167 |
| Doublons neutralisés | 3 |
| Échecs | 0 |
| Questions sémantiques | 17 |
| Questions admissibles au top 1 | 15 |
| Rappel sémantique top 1 | 86,666667 % |
| Rappel sémantique top 5 | 100 % |
| Diagnostic de plomberie top 1 / top 5 | 100 % / 100 % |
| Temps total avec questions | 1,383253 s |
| Consolidation | 124,44 événements/s |
| Rappel p95 | 4,6938 ms |
| Mémoire SQLite avec WAL/SHM | 10 308 544 octets |
| File SQLite avec WAL/SHM | 4 488 696 octets |

## Exécution à 100 faits sources

Configuration équivalente à l'option « 100 · rude » de l'interface, avec
100 questions au maximum.

| Mesure | Valeur |
|---|---:|
| Faits sources de base | 100 |
| Contradictions ajoutées | 8 |
| Valeurs calculées | 675 |
| Soumissions totales | 795 |
| Événements uniques | 783 |
| Doublons neutralisés | 12 |
| Échecs | 0 |
| Questions sémantiques | 56 |
| Questions admissibles au top 1 | 48 |
| Rappel sémantique top 1 | 83,333333 % |
| Rappel sémantique top 5 | 94,642857 % |
| Diagnostic de plomberie top 1 / top 5 | 100 % / 100 % |
| Temps de mise en file | 0,584966 s |
| Temps de consolidation | 17,923732 s |
| Temps total avec questions | 18,940807 s |
| Consolidation | 43,69 événements/s |
| Rappel p50 / p95 / p99 | 5,8075 / 61,782345 / 110,910177 ms |
| Rappel maximal mesuré | 114,2448 ms |
| Mémoire SQLite avec WAL/SHM | 27 907 240 octets |
| File SQLite avec WAL/SHM | 5 496 456 octets |

Dans les deux exécutions, la file s'est vidée, tous les travaux uniques ont été
terminés, les doublons ont eu un effet nul, les calculs sont restés `inferred`
et le répertoire temporaire a été supprimé.

## Interprétation honnête

Le passage de 167 à 783 événements uniques fait diminuer le débit de
consolidation de 124,44 à 43,69 événements/s. Cela confirme la limite déjà
connue : `MemoryEngine.observe` reconstruit encore des preuves et certains
agrégats globaux après chaque observation. Ce coût doit devenir incrémental
avant un test à un million de faits.

Le 100 % du diagnostic de plomberie confirme que les marqueurs exacts ont été
routés vers les épisodes attendus. Ce n'est pas un score de compréhension. Sur
les 56 questions sémantiques du run à 100 faits, le top 1 atteint 83,333333 %
et le top 5 94,642857 %. Ces résultats contrôlés montrent à la fois une capacité
de rappel et des erreurs réelles; ils ne signifient pas que l'agent répondrait
correctement à n'importe quelle question historique.

Les unités inconnues restent conservées dans le fait normalisé comme valeurs
opaques (`calculable: false`, avec une raison de saut). Seule leur dérivation
est ignorée : elles ne sont ni rejetées, ni devinées, ni converties. La prochaine
difficulté mesurable sera d'importer un sous-ensemble historique réel, de
conserver plusieurs sources et d'évaluer des formulations non générées par
l'oracle.
