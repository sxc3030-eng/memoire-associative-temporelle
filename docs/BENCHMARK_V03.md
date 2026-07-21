# Benchmark du pipeline v0.3

Mesures effectuées le 21 juillet 2026 sur le poste local Windows, avec `python scripts/benchmark_pipeline.py`. Les bases sont temporaires et chaque série contient environ 20 % de rejeux idempotents. Les souvenirs du benchmark sont fiables (`observed`) afin de mesurer le vrai coût de consolidation des preuves.

| Mesure | 100 soumissions | 500 soumissions |
|---|---:|---:|
| Souvenirs uniques | 81 | 401 |
| Doublons évités | 19 (19,0 %) | 99 (19,8 %) |
| Échecs | 0 | 0 |
| Temps de mise en file | 0,114 s | 0,428 s |
| Injection p50 / p95 | 1,016 / 1,821 ms | 0,684 / 1,505 ms |
| Temps total de vidange | 0,339 s | 3,278 s |
| Débit moyen de consolidation | 239,27/s | 122,34/s |
| Rappel concurrent p50 / p95 / p99 | 0,554 / 0,945 / 2,448 ms | 0,538 / 0,840 / 0,987 ms |
| Erreurs de lecture | 0 | 0 |
| Mémoire, DB + WAL + SHM | 5 597 232 octets | 8 996 168 octets |
| File, DB + WAL + SHM | 3 884 976 octets | 4 460 120 octets |

## Ce que le test démontre

- L'acceptation durable reste courte et indépendante de la consolidation.
- Le lecteur continue à rappeler un souvenir existant pendant les transactions d'écriture, sans erreur dans ces séries.
- Le dédoublonnage idempotent retire les rejeux avant qu'ils renforcent la mémoire.
- Les connexions lecteur et écrivain sont réellement distinctes.

Les tests automatisés complètent ce benchmark : crash après écriture avant acquittement, bail de worker, ordre temporel après erreur transitoire, double requête HTTP, oubli puis relivraison, import JSON renommé, lecture pendant une transaction SQLite ouverte, création atomique d'un run synthétique et nettoyage serveur sans client. Ils vérifient aussi qu'un nettoyage interrompu est repris et qu'un bilan persiste après la purge des souvenirs et tickets du run.

Le test visible ne dépend donc pas de la durée de vie de l'onglet. Son run et tous ses tickets sont enregistrés ensemble, puis le serveur possède le nettoyage. Les mutations de la file et de l'écrivain mémoire utilisent `synchronous=FULL`; les lectures continuent par une connexion distincte. Cette garantie réduit le risque de contamination par les données synthétiques, mais ne transforme pas le prototype en système de production.

## Ce que le test ne démontre pas

Le débit de consolidation baisse déjà lorsque la série grandit. La cause connue est `_refresh_aggregates()`, qui recalcule encore globalement les continuations, ainsi que la reconstruction de l'épisode touché. Le pipeline masque ce coût à l'injecteur, mais ne le supprime pas. Ces résultats ne doivent donc pas être extrapolés à un million ou un milliard d'éléments.

Les tailles sont l'empreinte instantanée des fichiers ouverts, journaux WAL/SHM compris. Elles incluent le schéma, le souvenir repère et de l'espace réutilisable; elles ne représentent pas un coût marginal pur par souvenir. Le registre persistant des runs conserve aussi un petit bilan après leur nettoyage, mais pas leurs textes synthétiques ni leurs tickets.

Enfin, ce benchmark ne prouve aucune réduction de paramètres d'un modèle IA. Il valide seulement l'infrastructure permettant de tester cette hypothèse. La preuve demandera de comparer, sur les mêmes tâches, un grand modèle seul, un petit modèle seul, un petit modèle avec RAG et le même petit modèle avec cette mémoire, tout en mesurant exactitude, raisonnement, données d'entraînement, paramètres, tokens de contexte, latence et stockage total.

## Prochaine optimisation mesurable

1. Remplacer le rafraîchissement global par des deltas sur les continuations touchées.
2. Conserver des compteurs de file incrémentaux au lieu de scanner les tickets.
3. Définir rétention, compaction et archivage du journal d'injection.
4. Mesurer le coût marginal après checkpoint pour plusieurs tailles croissantes.
5. Refaire les séries à 1 000, 10 000 puis 100 000 événements avant tout choix PostgreSQL, graphe ou vectoriel.
