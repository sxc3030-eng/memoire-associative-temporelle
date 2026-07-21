# Protocole de benchmark MAT-LM

Ce protocole sépare trois questions différentes : le modèle respecte-t-il le
contrat, produit-il la bonne valeur, et sa réponse est-elle fidèle au sens des
preuves ? Un seul score agrégé ne permet pas de répondre honnêtement aux trois.

## 1. Séparation des données

Utiliser trois graines et trois rôles distincts :

- `train` : spécialisation des poids ;
- `dev` : perte de validation, choix des hyperparamètres et débogage ;
- `test scellé` : exécuté une fois après gel du modèle et du protocole.

Le fichier actuel `matlm-dev-v8.jsonl` est un jeu **dev**, car le programme
d'entraînement peut calculer sa perte et influencer les décisions. Il ne doit
donc pas servir de test officiel. Un test scellé d'au moins 900 mondes, soit 100 par
famille, doit être généré avec une troisième graine après le gel du modèle.

Avant tout entraînement, publier les empreintes SHA-256 des JSONL, manifestes,
générateur, gabarits, contrat, tokenizer, modèle de base et commit. L'audit doit
refuser tout chevauchement exact entre train, dev et test pour :

- tous les identifiants, y compris `example_id`, `request_id`, identifiants de
  calcul et appels d'outil ;
- `synthetic_world` et `generator_sha256` ;
- chaque `evidence_id` ;
- chaque texte de preuve ;
- le texte de réponse cible, le dernier message assistant et l'objet cible
  canonique.

Les invites système et les structures JSON communes sont des gabarits
contractuels partagés et ne sont pas des faits. Elles sont inventoriées mais ne
doivent jamais être présentées comme une preuve de généralisation factuelle.

### Audit v8 du 21 juillet 2026

| Dimension | Train | Dev | Chevauchement |
|---|---:|---:|---:|
| Exemples / mondes uniques | 2 250 | 270 | 0 |
| Tous les identifiants imbriqués | 7 250 | 870 | 0 |
| `evidence_id` | 4 750 | 570 | 0 |
| Textes de preuve SHA-256 | 4 750 | 570 | 0 |
| Objets cibles SHA-256 | 2 250 | 270 | 0 |
| Derniers messages assistant SHA-256 | 2 250 | 270 | 0 |
| Textes `target.answer` distincts | 2 250 | 270 | 0 |

Empreintes des artefacts audités :

- train JSONL : `e79bd5ff4f27c572c0a6d7f122a51dd3ba9b43c7eb9f21e6f1dd56bcb76625c4` ;
- dev JSONL : `aa2d116ecce079e9a9e33e2df18d9bfdf8e30b3084116346fafc0a711e389a63` ;
- générateur train : `9413ade8c18eae9ac5a964d34169b5092c89aa1a0b84da82cdfd69dd21da8828` ;
- générateur dev : `9019a0913d272d956b7e3f8fce2a4b5a18c8e82c53d165487ecab7b6a7f940fb`.

Les neuf dimensions contrôlées ont un chevauchement train/dev nul. Les objets
cibles, derniers messages assistant et textes `target.answer` sont tous
uniques dans chaque split. Le contrôle doit néanmoins être rejoué sur tout
futur test scellé.

L'audit v1 avait auparavant rejeté les artefacts : 23 empreintes de texte cible
communes touchaient 312 lignes train et 60 lignes dev :

- `causal_uncertainty` : un gabarit générique identique, 250 train et 30 dev ;
- `date_age_arithmetic` : 22 résultats numériques répétés, 62 train et 30 dev.

Il ne s'agissait pas d'un chevauchement de monde, d'entité, de preuve ou de
fait, mais l'exactitude textuelle aurait pu récompenser la mémorisation du
gabarit. Le générateur v2 a lié ces réponses aux entités et dates du monde.
Les versions suivantes ont aligné exactement les invites d'entraînement et
d'inférence, remplacé le schéma verbeux par un gabarit JSON compact et raccourci
les réponses sans retirer d'information. En v8, aucun des 2 520 exemples ne
dépasse 1 024 tokens. Un nonce artificiel reste interdit : la différence doit
porter une information utile à la réponse.

## 2. Bras comparés

Chaque cas est exécuté avec la même question, les mêmes limites et un seul
modèle chargé à la fois :

1. modèle général, capsule vide ;
2. même modèle général, capsule complète ;
3. MAT-LM, capsule complète ;
4. MAT-LM, capsule complète et calculateur déterministe.

Pour chaque monde, préparer des variantes appariées : capsule complète, preuve
essentielle retirée, distracteurs, contradiction documentée, ordre des preuves
permuté et capsule d'un autre monde. Les identifiants opaques sont recalculés
pour le cas courant afin d'empêcher leur mémorisation.

Température, graine, longueur maximale, gabarit et contraintes restent fixes.
Les mesures froides et chaudes sont séparées et l'ordre des bras est
contrebalancé entre lots afin de limiter les effets thermiques.

### Comparaison au grand modèle local

Le premier témoin de grande taille déjà présent sur la machine est
`qwen2.5:14b-instruct-q4_0` (empreinte Ollama courte `5449194ff803`, 14,8
milliards de paramètres annoncés). Il ne sera jamais chargé en même temps que
Granite. La comparaison finale ajoute deux bras aux bras Granite :

1. Qwen 14B sans preuve externe ;
2. Qwen 14B avec exactement la même capsule que MAT-LM.

La matrice sépare ainsi trois effets : le gain de la mémoire, le gain du LoRA
et le gain brut de taille du modèle. Les questions, limites de génération,
ordre des preuves et graines restent identiques. Le score sémantique (réponse,
abstention, preuves et calcul) est séparé du score de respect du contrat JSON,
afin qu'un grand modèle ne perde pas artificiellement pour une différence de
mise en forme.

Le LoRA v0.8 est déjà le bras « fine-tuné ». Il apprend le langage du contrat
et les opérations de lecture; les faits du test restent dans la mémoire
externe et ne doivent jamais entrer dans l'entraînement. Un second fine-tuning
n'est justifié que si une famille d'opérations échoue sur le jeu scellé, jamais
pour mémoriser les réponses de ce jeu.

Le pilote Qwen de neuf cas a été exécuté sans client HTTP par
[`benchmark_ollama_heldout.py`](../scripts/benchmark_ollama_heldout.py). La
commande reproductible, la séparation contrat/contenu et les garde-fous sont
décrits dans [le benchmark Ollama CLI](OLLAMA_CLI_HELDOUT_BENCHMARK.md). Le
résultat reproductible est 5/9 au contrat brut, 7/9 aux preuves exactes et 3/9 à toutes
les ancres, contre 9/9 pour MAT-LM vérifié. Cette comparaison reste un pilote
dev et non le benchmark scellé.

Avant l'exécution, le protocole gèle les critères suivants : exactitude et
abstention sur au moins 900 cas scellés, taux d'hallucination, fidélité des
preuves, latences p50/p95, débit, pic mémoire et taille disque. Le succès de
l'hypothèse sera déclaré si MAT-LM 2B reste à moins de cinq points d'exactitude
du témoin 14B sur les tâches de mémoire, sans augmenter les hallucinations de
plus de deux points, avec un artefact de modèle nettement plus petit.

## 3. Trois familles de métriques

### Métriques contractuelles, déterministes

- JSON analysable et schéma exact ;
- `request_id` correct et aucun champ supplémentaire ;
- zéro `evidence_id` inventé et respect du nombre maximal ;
- cohérence entre preuve obligatoire, réponse et abstention ;
- calcul déclaré conforme au contrat et limites de longueur respectées.

Une sortie invalide reste invalide : elle ne reçoit pas un score de contenu.

### Métriques exactes, déterministes

- valeur ou ensemble de valeurs exactes : code, état, date, âge, rôle ;
- résultat de calcul réexécuté indépendamment ;
- précision, rappel et F1 des preuves attendues ;
- choix exact de la version corrigée ou de la source prioritaire ;
- exactitude des abstentions, faux refus et réponses non soutenues.

L'égalité de prose complète est secondaire. Les valeurs sont extraites dans
des champs structurés ou comparées avec une normalisation publiée.

### Métriques sémantiques, secondaires

- réponse impliquée par les seules preuves citées ;
- rôles correctement distingués ;
- causalité présentée comme certaine, possible ou non établie au bon niveau ;
- contradiction résumée sans effacer la trace remplacée.

Pour le premier résultat officiel, utiliser deux évaluateurs humains aveugles
au bras, un arbitrage des désaccords et publier l'accord inter-évaluateurs. Un
juge local automatisé pourra être ajouté ensuite, mais sa version, son invite
et un échantillon audité devront être figés ; son score ne remplacera jamais
les métriques exactes.

## 4. Analyse et seuil de preuve

Publier les résultats par famille et une macro-moyenne donnant le même poids
aux neuf familles. Comparer les bras sur les mêmes items avec intervalle de
confiance bootstrap apparié à 95 % et test de McNemar pour les décisions
binaires. Rapporter aussi latence p50/p95, débit, RAM/VRAM maximale, taille de
capsule et longueur de sortie.

Une conclusion favorable exige simultanément :

- aucune collision de données au contrôle préliminaire ;
- zéro identifiant de preuve inventé ;
- gain mémoire sur le modèle général, bras 2 contre bras 1 ;
- gain de spécialisation, bras 3 contre bras 2, avec intervalle apparié ;
- gain de calcul, bras 4 contre bras 3, sans dégradation des autres familles ;
- pas de hausse importante des réponses non soutenues ou des faux refus.

Le JSONL test et ses cibles restent côté évaluateur. Après un résultat test, une
modification du modèle, du prompt, du seuil ou du correcteur crée une nouvelle
version du protocole et exige un nouveau split scellé.
