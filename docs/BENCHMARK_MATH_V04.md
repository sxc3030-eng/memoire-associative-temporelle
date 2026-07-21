# Benchmark du calculateur v0.4

Mesures effectuées le 21 juillet 2026 sur le poste local Windows, avec Python 3.12.13 et le registre `math-core-v1`. Les 100 000 expressions couvrent dix familles déterministes et leurs résultats attendus sont produits par un chemin Python indépendant du parseur.

## Exécution de vitesse

Commande :

```bash
python scripts/benchmark_math.py --count 100000 --warmup 2000
```

| Mesure | Résultat |
|---|---:|
| Expressions correctes | 100 000 / 100 000 |
| Divergences / erreurs | 0 / 0 |
| Durée totale | 1,945 s |
| Débit moyen | 51 425,53 expressions/s |
| Latence p50 / p95 / p99 | 0,0169 / 0,0278 / 0,0309 ms |
| Latence maximale échantillonnée | 0,1141 ms |
| Sondes de sécurité rejetées | 5 / 5 |
| Écritures dans la mémoire associative | 0 |

Le catalogue contient 41 fonctions réparties en 6 catégories. Sa représentation JSON canonique mesurée fait 13 021 octets. Le fichier source du moteur fait 29 018 octets et le catalogue d'exemples 2 678 octets; ces tailles de fichiers ne sont pas une mesure de la mémoire vive de Python.

Les latences sont échantillonnées avec un pas premier avec le cycle des dix familles : 33 334 appels pour la série de 100 000 et 47 620 pour celle d'un million. Elles couvrent donc toutes les familles au lieu de répéter un seul type d'expression.

## Passage à un million

Commande :

```bash
python scripts/benchmark_math.py --count 1000000 --warmup 5000
```

| Mesure | Résultat |
|---|---:|
| Expressions correctes | 1 000 000 / 1 000 000 |
| Divergences / erreurs | 0 / 0 |
| Durée totale | 19,620 s |
| Débit moyen | 50 967,78 expressions/s |
| Latence p50 / p95 / p99 | 0,0171 / 0,0279 / 0,0314 ms |
| Latence maximale échantillonnée | 0,2108 ms |
| Écritures dans la mémoire associative | 0 |

Le débit reste du même ordre que pour 100 000 expressions. Le script ne conserve pas les résultats et ne crée aucune base de mémoire; cette série vérifie donc le calcul à grande répétition, pas une ingestion d'un million de souvenirs.

## Exécution avec suivi des allocations

Commande :

```bash
python scripts/benchmark_math.py --count 100000 --warmup 2000 --measure-memory
```

| Mesure | Résultat |
|---|---:|
| Expressions correctes | 100 000 / 100 000 |
| Durée sous `tracemalloc` | 7,429 s |
| Débit sous `tracemalloc` | 13 460,28 expressions/s |
| Allocations Python courantes à la fin | 1 077 433 octets |
| Pic d'allocations Python suivi | 1 104 744 octets |
| Écritures dans la mémoire associative | 0 |

`tracemalloc` ralentit fortement l'exécution. Son débit et ses latences ne doivent pas être comparés directement au test de vitesse. Le pic indiqué couvre les allocations Python suivies pendant la boucle, pas toute la mémoire résidente du processus, de l'interpréteur ou du système.

## Ce que la mesure établit

- Le moteur retourne les résultats attendus pour les familles générées par ce benchmark.
- La mémoire n'augmente pas avec le nombre de calculs, puisque le benchmark n'instancie ni base mémoire ni file d'injection.
- Les tentatives simples d'import, d'accès à un attribut, de compréhension, d'ouverture de fichier et de fonction anonyme sont refusées.
- Le coût du catalogue reste lié au nombre de règles, pas au nombre de réponses calculées.

## Ce que la mesure n'établit pas

- Les 100 000 cas ne représentent pas toutes les mathématiques ni tous les cas numériques limites.
- Les sondes adversariales complètent les tests unitaires, mais ne constituent pas une preuve formelle de sécurité.
- Le débit d'un cœur local ne s'extrapole pas directement à un milliard d'expressions.
- Aucun gain de paramètres ou de données d'entraînement d'un modèle IA n'est démontré ici.

À partir du débit observé de 50 968 expressions/s, une extrapolation purement arithmétique donne environ 5,45 heures pour un milliard d'expressions séquentielles. Ce n'est pas une mesure à un milliard : une expérience réelle à cette échelle devra réutiliser les arbres validés, traiter par lots, répartir le travail et mesurer séparément calcul, mémoire, orchestration et stockage.
