# MAT-LM-2B : petit modèle dédié à la mémoire

## Objectif

MAT-LM n'essaie pas de graver tous les faits dans ses poids. Il apprend un
comportement : lire une capsule bornée, choisir les preuves utiles, relier des
faits, demander un calcul vérifiable, citer les identifiants utilisés et
s'abstenir lorsque la capsule ne suffit pas.

Le premier candidat utilise
[`ibm-granite/granite-3.3-2b-instruct`](https://huggingface.co/ibm-granite/granite-3.3-2b-instruct)
comme socle Apache-2.0. Le produit propre au projet est l'adaptateur LoRA, le
curriculum synthétique, le contrat mémoire, les validateurs et le harnais
d'évaluation. Il ne s'agit donc ni d'un préentraînement de deux milliards de
paramètres depuis zéro, ni d'une simple copie du modèle de base.

## Séparation des responsabilités

```mermaid
flowchart LR
    Q["Question"] --> H["Memory Hub"]
    H --> C["Capsule JSON bornée"]
    C --> M["MAT-LM-2B + adaptateur LoRA"]
    M --> V["Validation des preuves"]
    V --> K["Calculateur déterministe"]
    K --> R["Réponse acceptée ou refusée"]
```

- Les faits restent dans les espaces `private`, `shared` ou `reference`.
- Le modèle ne peut citer que les `evidence_id` présents dans la capsule.
- Une réponse sans preuve est refusée lorsque la preuve est obligatoire.
- Le texte d'un calcul produit par le modèle n'est jamais cru : l'expression
  est réexécutée par le calculateur borné.
- Une réponse générée ne retourne jamais automatiquement dans l'apprentissage.

Le contrat complet est décrit dans
[`MEMORY_NATIVE_LLM_CONTRACT.md`](MEMORY_NATIVE_LLM_CONTRACT.md).

## Curriculum sans fuite

Le curriculum par défaut crée 2 250 mondes fictifs et déterministes, répartis
également entre neuf familles : rappel direct, relation à plusieurs sauts,
âge et dates, distinction des rôles, prudence causale, résolution de
contradictions, rejet de distracteurs, abstention et choix de provenance.

Les identités, faits et sources du test scientifique sont interdits. Un audit
cherche explicitement toute collision. Le jeu d'évaluation utilise une autre
graine et ses identifiants ne peuvent pas apparaître dans l'entraînement.

```powershell
python scripts/build_memory_native_curriculum.py `
  --seed 20260721 --count 2250 `
  --forbidden-corpus examples/science-biographies-v1.json `
  --output training-data\matlm-train-v8.jsonl `
  --manifest-output training-data\matlm-train-v8.manifest.json

python scripts/build_memory_native_curriculum.py `
  --seed 20260722 --count 270 `
  --forbidden-corpus examples/science-biographies-v1.json `
  --output training-data\matlm-dev-v8.jsonl `
  --manifest-output training-data\matlm-dev-v8.manifest.json
```

## Entraînement local

La cible locale actuelle est une Intel Arc B570 de 10 Go. Le mode prévu est
QLoRA NF4 lorsque les noyaux `bitsandbytes` sont disponibles; le repli est un
LoRA BF16 limité aux projections d'attention, avec lot 1, checkpointing des
activations et séquences courtes. Le script ne pousse rien vers un hub et ne
contacte aucun service d'inférence.

Validation à blanc :

```powershell
python scripts/train_matlm.py `
  --train-jsonl training-data\matlm-train-v8.jsonl `
  --eval-jsonl training-data\matlm-dev-v8.jsonl `
  --output-dir training-runs\matlm-plan `
  --cache-dir D:\MAT-LM\hf-cache `
  --mode bf16-lora --sequence-length 1024 --max-steps 1 --dry-run
```

Premier essai réel :

```powershell
D:\MAT-LM\.venv\Scripts\python.exe scripts\train_matlm.py `
  --train-jsonl training-data\matlm-train-v8.jsonl `
  --eval-jsonl training-data\matlm-dev-v8.jsonl `
  --output-dir training-runs\MAT-LM-2B-v0.8-pilot-300 `
  --base-model D:\MAT-LM\models\granite-3.3-2b-instruct `
  --mode bf16-lora --fallback none `
  --sequence-length 1024 --gradient-accumulation-steps 1 `
  --max-steps 300 --learning-rate 0.0001
```

Le manifeste contient l'empreinte des données, les hyperparamètres, les
versions logicielles, le mode réellement utilisé, la fraction de paramètres
entraînés et les mesures du run. Un dossier de sortie non vide n'est jamais
écrasé.

## Évaluation

La comparaison charge toujours un seul modèle à la fois :

1. socle général sans mémoire;
2. même socle avec capsule mémoire;
3. MAT-LM avec la même capsule;
4. MAT-LM avec capsule et calculateur vérifié.

Les mesures comprennent l'exactitude, le JSON valide, les preuves inventées,
les abstentions, les calculs réexécutés, la latence et la mémoire maximale.
Les neuf questions scientifiques locales restent entièrement hors du
curriculum et ne constituent qu'un test de fumée. Les benchmarks publics
seront téléchargés et versionnés séparément après validation de leurs licences.

Le pilote v0.8 et ses limites sont publiés dans
[`MAT_LM_PILOT_RESULTS.md`](MAT_LM_PILOT_RESULTS.md).

## Interroger l'adaptateur

Une fois un adaptateur produit :

```powershell
D:\MAT-LM\.venv\Scripts\python.exe scripts\ask_matlm.py `
  --adapter D:\MAT-LM\adapter `
  --base-model D:\MAT-LM\models\granite-3.3-2b-instruct `
  --load-mode bf16 `
  --capsule capsule.json
```

La sortie standard contient uniquement un objet `memory-native-answer-v1`
validé. Une génération brute invalide n'est ni publiée ni mémorisée.
