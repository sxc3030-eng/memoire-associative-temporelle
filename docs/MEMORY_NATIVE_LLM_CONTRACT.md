# Contrat d’un petit modèle natif de la mémoire

## But

Le modèle ne reçoit pas directement une base SQLite, un historique de conversation ou des instructions cachées dans un souvenir. Il reçoit une **capsule JSON bornée**, répond avec un **objet JSON vérifiable**, puis le programme accepte ou refuse cette réponse avant tout autre usage.

Ce contrat est neutre : un adaptateur local peut utiliser n’importe quel moteur de modèle, tant qu’il respecte les deux schémas. Aucun modèle et aucune donnée externe ne sont téléchargés par cette couche.

Implémentation : `src/memory_agent/native_llm_contract.py`.

## Entrée : `memory-native-capsule-v1`

```json
{
  "schema_version": "memory-native-capsule-v1",
  "request_id": "science-001",
  "question": "En quelle année le radium a-t-il été annoncé ?",
  "evidence": [
    {
      "evidence_id": "science:curie:radium",
      "text": "Marie et Pierre Curie annoncèrent la découverte du radium en 1898.",
      "space": "reference",
      "status": "verified",
      "confidence": 1.0,
      "temporal_context": "1898",
      "tags": ["chimie", "radium"]
    }
  ],
  "constraints": {
    "evidence_required": true,
    "allow_calculations": true,
    "max_answer_characters": 4000,
    "max_evidence_ids": 8,
    "max_calculations": 4
  }
}
```

Principes :

- `request_id` relie sans ambiguïté la question à sa réponse.
- `evidence_id` est opaque et unique dans la capsule. Le modèle ne peut citer que ces identifiants.
- `space` distingue les souvenirs privés, partagés et les références.
- `status` exprime la nature de la preuve; une génération du modèle n’est pas un statut accepté.
- `temporal_context` conserve une date ou une période sans imposer un calendrier particulier.
- `constraints` borne la réponse avant l’inférence.
- Les clés inconnues, doublons, nombres non finis et contenus hors limite sont refusés.

La constante `CAPSULE_JSON_SCHEMA` est un schéma JSON Draft 2020-12 avec `additionalProperties: false`. `validate_capsule()` ajoute les invariants que le schéma seul exprime mal, notamment l’unicité globale des preuves.

## Sortie : `memory-native-answer-v1`

```json
{
  "schema_version": "memory-native-answer-v1",
  "request_id": "science-001",
  "answer": "Le radium a été annoncé en 1898.",
  "confidence": 0.98,
  "evidence_ids": ["science:curie:radium"],
  "calculations": [],
  "abstention": {
    "abstained": false,
    "reason": "none",
    "missing_information": []
  }
}
```

Un calcul auditable a cette forme :

```json
{
  "calculation_id": "calc-1",
  "expression": "1903 - 1898",
  "reported_result": "5",
  "unit": "ans",
  "evidence_ids": ["science:curie:radium"]
}
```

`calculations` n’est pas une chaîne de pensée. C’est uniquement une demande de vérification courte. **`reported_result` demeure un texte non fiable produit par le modèle, même après `validate_answer()`.** Le validateur contrôle le format, les bornes et les preuves citées; il n’exécute jamais l’expression. Seul le moteur mathématique borné peut recalculer `expression`, comparer le résultat et produire ensuite une observation `executed` distincte. Le résultat rapporté ne doit jamais être mémorisé directement.

Une abstention explicite utilise une raison parmi :

- `insufficient_evidence`;
- `contradictory_evidence`;
- `out_of_scope`;
- `unsafe_request`;
- `invalid_capsule`.

Lors d’une abstention, la confiance dans la réponse vaut `0`. Sans abstention, `reason` vaut `none`, `missing_information` est vide et une réponse non vide est obligatoire.

## Validation déterministe

`validate_answer(response, capsule)` refuse notamment :

- un `request_id` différent;
- une clé supplémentaire comme `chain_of_thought`;
- un identifiant de preuve absent de la capsule;
- une preuve de calcul absente de la liste principale des preuves citées;
- un calcul lorsque la capsule l’interdit;
- une réponse sans preuve lorsque `evidence_required` est actif;
- une réponse ou une liste dépassant les limites demandées;
- une abstention ambiguë.

Une sortie validée reste une **réponse générée**. Elle ne revient jamais automatiquement dans l’apprentissage. Une observation extérieure, une exécution vérifiée ou une confirmation distincte reste nécessaire.

## Passage depuis le `MemoryHub`

L'adaptateur `matlm_bridge.py` convertit chaque résultat rappelé en preuve atomique :

1. produire un `evidence_id` stable à partir de l’espace, de l’épisode et de l’empreinte du résultat;
2. copier seulement le texte utile et la provenance autorisée;
3. traduire la politique du hub vers `private`, `shared` ou `reference`;
4. limiter et dédupliquer avant de construire la capsule avec `build_capsule()`;
5. garder toute correction et toute réponse attendue hors de la capsule.

Cette adaptation est maintenant intégrée au serveur v0.8. Elle reste aussi
testable indépendamment de l'interface et du moteur : le serveur ne transmet
que les espaces autorisés, borne la capsule et refuse toute sortie hors contrat.

## Comparaison à trois bras, un modèle à la fois

`src/memory_agent/native_llm_benchmark.py` définit un harnais séquentiel :

1. `baseline` — petit modèle général, preuves retirées;
2. `memory` — le même modèle général avec la capsule complète;
3. `specialized` — modèle dédié à ce contrat avec la même capsule complète.

Le fournisseur local ouvre une session dans un bloc de contexte. Ce modèle doit être fermé ou déchargé avant l’ouverture du suivant. Le harnais refuse aussi deux exécutions simultanées sur la même instance.

Chaque bras produit `ok`, `invalid_output` ou `model_error`, avec sa durée. L’échec d’un bras n’empêche pas les suivants, et la sortie brute invalide n’est pas conservée par défaut.

## Plan d’intégration aux benchmarks reconnus

### 1. Infrastructure de référence

Brancher plus tard un adaptateur sur [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness). Sa documentation officielle décrit les tâches génératives et à choix multiple ainsi que l’ajout de tâches personnalisées. Les révisions, paramètres de génération, gabarits et empreintes de fichiers devront être inscrits dans chaque rapport.

Aucun téléchargement n’est effectué maintenant. Chaque corpus devra faire l’objet d’une activation explicite et d’une vérification de licence.

### 2. Capacités générales

- [MMLU-Pro](https://arxiv.org/abs/2406.01574) : connaissances et raisonnement multi-domaines plus exigeants. Il mesure d’abord la capacité générale; la bonne option reste dans l’évaluateur, jamais dans la capsule.
- [GPQA](https://arxiv.org/abs/2311.12022), en commençant par le sous-ensemble Diamond : raisonnement scientifique difficile. Le score principal est l’exactitude, complété par le taux de JSON valide et l’abstention.
- [GSM8K](https://arxiv.org/abs/2110.14168) : calculs verbaux. Les expressions courtes de `calculations` sont recalculées indépendamment; le texte de raisonnement libre n’est pas exigé.

Ces tests comparent la compétence des modèles, mais ne prouvent pas à eux seuls la valeur de la mémoire.

### 3. Valeur propre de la mémoire

- [HotpotQA](https://hotpotqa.github.io/) fournit des questions multi-sauts et des faits de support. Les paragraphes autorisés deviennent des preuves opaques; la réponse et les faits attendus restent uniquement côté évaluateur.
- Créer des variantes contrôlées : capsule complète, preuve essentielle retirée, preuves contradictoires et distracteurs. Cela mesure rappel, raisonnement, citation et abstention au lieu de la simple mémorisation du modèle.
- Conserver les neuf questions scientifiques locales comme test de fumée, mais publier leurs résultats séparément des scores officiels.

### 4. Mesures communes

Pour chaque item et chaque bras :

- score officiel du benchmark;
- taux de réponses JSON valides;
- exactitude des `evidence_ids` et rappel des faits de support lorsqu’ils existent;
- taux d’hallucination de preuve;
- exactitude des abstentions sur les capsules amputées ou contradictoires;
- validité des calculs après réexécution;
- latence, taille de capsule, longueur de sortie et mémoire maximale du processus;
- résultat froid et résultat chaud consignés séparément.

### 5. Règles d’équité

- même modèle général et mêmes paramètres pour `baseline` et `memory`;
- même question, seul le contenu mémoire change;
- ordre fixé et un seul modèle chargé à la fois;
- température déterministe et graine fixe lorsque le moteur les permet;
- clés de correction, réponses attendues et split de test exclus de l’entraînement et de la mémoire;
- modèle spécialisé évalué sur un split jamais utilisé pour sa spécialisation;
- résultats par item conservés pour éviter qu’une moyenne cache les refus ou erreurs de format.

Le premier jalon réel sera un petit sous-ensemble figé de chaque famille, suivi du jeu complet seulement après validation du format et des métriques.
