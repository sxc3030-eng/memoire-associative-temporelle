# Benchmark multi-IA v0.6 — passage de fumée Ollama

## Portée

Ce rapport est un contrôle de câblage, pas une mesure générale d'intelligence.
Il pose une seule question historique aux 12 empreintes Ollama installées, une
fois sans mémoire puis une fois avec la capsule produite par le `MemoryHub`.

Les modèles sont chargés strictement l'un après l'autre, du plus petit au plus
grand. Après les deux réponses, le banc demande à Ollama de libérer le modèle
avant de passer au suivant. À la fin du passage, `/api/ps` ne signalait aucun
modèle chargé. Aucun modèle n'a été téléchargé, créé, supprimé ou remplacé.

Question : « Les travaux de quels naturalistes sur la sélection naturelle
furent-ils présentés ensemble à la Linnean Society, et quand ? »

La capsule neutre mesurait 9 942 caractères; l'adaptateur a transmis 4 623
caractères de faits et de sources lisibles au modèle. La grille de correction
n'était pas dans la capsule ni dans le prompt.

## Résultats

| Modèle | Type | Sans mémoire | Avec mémoire | Latence sans / avec | État |
|---|---:|---:|---:|---:|---|
| `deepseek-coder:1.3b` | texte | 0 % | 0 % | 2 066 / 1 388 ms | terminé |
| `qwen2.5-coder:1.5b-base` | base | 0 % | **100 %** | 3 408 / 1 639 ms | terminé |
| `llama3.2:latest` | texte | 100 % | 100 % | 4 339 / 1 252 ms | terminé |
| `mistral:7b-instruct-q4_0` | texte | 0 % | 0 % | 5 416 / 1 661 ms | terminé |
| `llama3.1:8b` | texte | 100 % | 100 % | 6 670 / 2 289 ms | terminé |
| `llama3.2-vision:11b` | vision | — | — | — | HTTP 500 sur les deux essais |
| `gemma3:12b` | texte | 100 % | 100 % | 18 615 / 4 898 ms | terminé |
| `qwen2.5:14b-instruct-q4_0` | texte | 100 % | 100 % | 17 822 / 7 095 ms | terminé |
| `klara:latest` | texte | 100 % | 100 % | 14 841 / 5 486 ms | terminé |
| `forge-mythos-pro:latest` | texte | 100 % | 100 % | 19 292 / 12 329 ms | terminé |
| `gemma3:12b-it-qat` | texte | 100 % | 100 % | 12 824 / 3 444 ms | terminé |
| `deepseek-coder:33b` | texte | — | — | > 180 000 ms par essai | délai dépassé deux fois |

Parmi les 10 modèles ayant produit les deux réponses, 7/10 ont réussi sans
mémoire et 8/10 avec mémoire, soit 70 % contre 80 % sur cette unique question.
Le gain observé vient du petit Qwen Coder Base. Il est réel pour cet essai mais
ne permet aucune extrapolation : plusieurs modèles connaissaient déjà le fait,
et deux petits modèles orientés code n'ont pas suivi suffisamment les preuves.

## Interprétation prudente

- le chemin complet inventaire → rappel → capsule → adaptateur → modèle → score
  fonctionne ;
- la déduplication par empreinte évite de compter deux fois les alias `latest` ;
- un petit modèle de base a répondu correctement seulement avec la mémoire ;
- une question facile produit un effet plafond chez sept modèles ;
- le modèle vision nécessite un diagnostic séparé de son erreur locale ;
- le 33B est trop lent avec la limite actuelle et ne doit pas bloquer le reste
  du protocole ;
- le prochain résultat utile doit couvrir les neuf questions, notamment âge,
  contemporanéité, rôles multiples, causalité incertaine et abstention.

Le rapport machine complet est conservé dans
`reports/ollama-all-models-smoke-v06.json`.
