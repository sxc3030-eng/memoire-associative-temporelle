# Défi scientifique v0.6 — Qwen 1,5B Base

## But du passage

Ce passage mesure un seul petit modèle local, `qwen2.5-coder:1.5b-base`, sur
les neuf questions tenues à l'écart du corpus scientifique. Chaque question
est posée deux fois : sans mémoire, puis avec une capsule construite uniquement
à partir des affirmations et de leurs sources. La grille de correction n'est
jamais incluse dans le prompt.

Ce test est un banc interne ciblé. Il ne constitue pas encore un benchmark
officiel généraliste.

## Résultat strict

| Mesure | Sans mémoire | Avec mémoire | Écart |
|---|---:|---:|---:|
| Exactitude stricte | 0,0 % | 22,2 % | +22,2 points |
| Abstention | 44,4 % | 0,0 % | -44,4 points |
| Hallucination interdite | 11,1 % | 11,1 % | 0 point |
| Latence moyenne | 1 827 ms | 2 189 ms | +363 ms |

Les 18 requêtes ont terminé sans erreur. Le modèle a ensuite été libéré de la
mémoire vive par Ollama.

## Ce que le test révèle

La capsule améliore nettement l'accès aux noms, dates et rôles pertinents, mais
le modèle Base de 1,5 milliard de paramètres ne transforme pas encore ces faits
de manière fiable. Il réussit complètement deux questions sur neuf seulement.
Il commet notamment une erreur de calcul sur l'âge de Marie Curie, interprète
mal la contemporanéité et n'abstient pas sur la question impossible concernant
un ordinateur en 1858.

Le résultat est utile précisément parce qu'il sépare deux capacités :

1. la mémoire retrouve les preuves pertinentes ;
2. le petit modèle doit encore apprendre à calculer, relier, distinguer les
   rôles et reconnaître qu'une réponse est impossible.

La prochaine comparaison pertinente utilisera exactement les mêmes capsules
et la même grille avec un petit modèle instruction-tuned, toujours un seul
modèle à la fois.

Le détail machine, y compris chaque réponse et chaque latence, est conservé
dans `reports/qwen2.5-coder-1.5b-science-v06.json`.
