# Comparer MAT-LM à Qwen avec Ollama CLI

Ce pilote passe à Qwen **exactement les neuf capsules** du pilote MAT-LM. Le
programme n'utilise aucun client HTTP et ne télécharge rien : il appelle
l'exécutable Ollama local par entrée standard, un seul tag à la fois. Ollama
peut gérer son propre service local, mais le harnais ne connaît aucune URL et
n'envoie rien à un service externe.

## Validation sans modèle

Sous PowerShell :

```powershell
py -3.13 scripts\benchmark_ollama_heldout.py `
  --dataset training-data\matlm-dev-v8.jsonl `
  --model qwen2.5:14b-instruct-q4_0 `
  --ollama-executable "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" `
  --limit 9 --dry-run
```

Le plan doit annoncer cette empreinte de sélection, déjà utilisée par MAT-LM :

```text
c884894291473583d82beaf8bd9d253d2c24ccded1f208ddfad33993e37c5d10
```

Le `dry-run` ne cherche même pas l'exécutable et ne lance aucun modèle.

## Pilote réel de neuf cas

Fermer d'abord toute autre génération locale, puis lancer :

```powershell
py -3.13 scripts\benchmark_ollama_heldout.py `
  --dataset training-data\matlm-dev-v8.jsonl `
  --model qwen2.5:14b-instruct-q4_0 `
  --ollama-executable "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" `
  --limit 9 `
  --case-timeout-seconds 180 `
  --total-timeout-seconds 3600 `
  --report reports\qwen2.5-14b-dev9-cli-v2.json
```

Avant la première question, le programme exécute seulement `ollama --version`
et `ollama show <tag>`. Si le tag n'est pas déjà installé, il s'arrête : aucun
`pull` implicite n'est permis. Après le dernier cas, il demande à Ollama de
libérer ce tag avec `ollama stop`.

Le harnais ajoute `--nowordwrap` à `ollama run`. Sans cette option, le client
Windows peut insérer des déplacements de curseur et des fragments répétés dans
la sortie standard pendant la génération, ce qui corrompt un JSON pourtant
correct à l'écran.

## Lire la comparaison

Comparer uniquement des rapports dont `dataset.selection_sha256` est
identique. Les scores répondent à des questions différentes :

- `contract_valid` : la sortie respecte exactement
  `memory-native-answer-v1` ;
- `answer_exact_normalized` : la prose est identique à la cible après une
  normalisation minimale ;
- `answer_anchor_recall` et `answer_anchors_all` : présence des identifiants
  `SYN-…`, valeurs `code/état-fictif-N`, dates ISO, âges, `JE_NE_SAIS_PAS` et
  décision initiale Oui/Non de la cible, sans publier ces ancres ;
- `content_core_exact` : prose normalisée, preuves et abstention sont exactes,
  même si un champ superflu invalide le contrat ;
- `content_all_exact` ajoute les calculs exacts ;
- `invented_evidence_ids` mesure les citations absentes de la capsule.

Le rappel d'ancres est plus équitable envers une paraphrase, mais **ce n'est pas
un score sémantique** : il ne prouve ni que les rôles sont bien attribués, ni que
deux codes cités sont classés dans le bon ordre. L'exactitude de prose reste
stricte et secondaire; l'épreuve officielle doit ajouter l'évaluation aveugle
décrite par le protocole.

Le rapport conserve seulement des compteurs, temps, empreintes SHA-256 et
erreurs neutralisées. Questions, preuves, réponses cibles et sorties du modèle
ne sont jamais écrites dans le rapport. Les identifiants de cas y sont hachés.

Ce lot de neuf cas est un test de plomberie, pas une preuve statistique. La
comparaison officielle doit être refaite sur le test scellé d'au moins 900 cas
décrit dans `MAT_LM_BENCHMARK_PROTOCOL.md`.

## Résultat local du 21 juillet 2026

Le témoin était le Qwen 2.5 Instruct Q4_0 déjà installé dans Ollama : 14,8
milliards de paramètres, contexte annoncé de 32 768 tokens. Il a reçu la même
sélection que MAT-LM, d'empreinte
`c884894291473583d82beaf8bd9d253d2c24ccded1f208ddfad33993e37c5d10`.

| Mesure | Qwen 14,8B non adapté |
|---|---:|
| Contrat JSON strict | 4/9 |
| `request_id` correct | 9/9 |
| Ensemble de preuves exact | 7/9 |
| Toutes les ancres factuelles présentes | 3/9 |
| Rappel moyen des ancres | 62,0 % |
| Abstention exacte | 3/9 |
| Calculs exacts | 8/9 |
| Réponse textuelle exactement identique | 0/9 |
| Preuve inventée | 0 |

Les neuf générations ont pris 77,42 secondes au total; la médiane par cas est
6,90 secondes. Ces temps incluent le chargement froid du premier cas et ne
sont pas directement comparables au harnais Transformers de MAT-LM.

Le score textuel exact de 0/9 pénalise toute paraphrase. Les scores de preuves
et d'ancres montrent que Qwen a souvent trouvé les bons éléments, mais il n'a
pas appris de façon fiable le contrat natif, les règles d'abstention ni toutes
les relations attendues. Ce pilote mesure donc une spécialisation à la mémoire,
pas l'intelligence générale des deux modèles.

Rapports assainis :

- résultat corrigé : `reports/qwen2.5-14b-dev9-cli-v2.json`, SHA-256
  `93425fce54ef319d53121f9241e55f439a533e1f0d2745ff0534a0f911f92011` ;
- première exécution invalidée par le word-wrap :
  `reports/qwen2.5-14b-dev9-cli.json`, SHA-256
  `c1b3e72158a6ba694f11db71ebeeebad508c13f0ecb7f2a8948a0f1aedb5a976`.

Le second fichier est conservé pour l'audit du banc, mais ses 0/9 ne doivent
jamais être présentés comme un résultat de Qwen.

## Variante fine-tunée

Un Qwen adapté hors ligne peut être évalué avec la même commande en remplaçant
seulement `--model` par son tag Ollama local. Le fichier Q4_0 installé dans
Ollama est toutefois un artefact d'inférence quantifié : notre pipeline ne le
fine-tune pas directement. Il faudrait un checkpoint entraînable distinct,
assez de mémoire et un nouvel adaptateur LoRA; aucun poids n'est téléchargé
implicitement.

Pour garder une comparaison valide, le fine-tuning utilise le split `train`,
le choix des réglages utilise `dev`, et ni les capsules ni les cibles du test
scellé ne doivent entrer dans les poids. Le grand Qwen non adapté reste le
témoin; le Qwen adapté constitue un bras distinct, pas un remplacement du
témoin.
