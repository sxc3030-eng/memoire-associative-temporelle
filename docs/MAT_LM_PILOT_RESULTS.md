# Résultats du pilote MAT-LM v0.8

Date : 21 juillet 2026. Ces résultats sont un **test de développement local**,
pas encore un benchmark officiel scellé.

## Configuration

- socle : Granite 3.3 2B Instruct, poids locaux et hors ligne ;
- adaptation : LoRA BF16 sur `q_proj`, `k_proj`, `v_proj`, `o_proj` ;
- paramètres entraînés : 4 259 840 sur 2 537 799 680, soit 0,168 % ;
- taille des poids LoRA : 17 081 704 octets ;
- entraînement : 300 pas, lot 1, séquences de 1 024 tokens, aucune
  troncature ;
- matériel : Intel Arc B570, BF16, un seul modèle chargé à la fois ;
- réseau et API d'inférence : désactivés.

La perte d'entraînement finale est 0,07550. Les pertes dev mesurées sont
0,03736 au pas 100, 0,02320 au pas 200 et 0,02272 au pas 300. Le pic XPU
alloué est 5 892 959 232 octets et le pic réservé 7 950 303 232 octets.

## Comparaison A/B appariée

Les deux bras reçoivent exactement les neuf mêmes capsules, une par famille,
avec l'empreinte de sélection
`c884894291473583d82beaf8bd9d253d2c24ccded1f208ddfad33993e37c5d10`.

| Bras | JSON/contrat valide | Preuves exactes | Tout exact |
|---|---:|---:|---:|
| Granite vierge + capsule | 0/9 | 0/9 | 0/9 |
| MAT-LM v0.8 avant recalcul | 9/9 | 9/9 | 8/9 |
| MAT-LM v0.8 + calculateur déterministe | 9/9 | 9/9 | 9/9 |

L'unique erreur avant recalcul concernait l'âge civil : le modèle avait lu
les deux bonnes dates et produit la bonne expression, mais annonçait 26 au
lieu de 27. Le calculateur borné a réexécuté `calendar_age`, corrigé le
résultat et revalidé l'objet complet. Aucune expression arbitraire n'est
exécutée.

## Comparaison au grand modèle local

Le même lot a ensuite été présenté, capsule comprise, à
`qwen2.5:14b-instruct-q4_0` dans Ollama. Le modèle local annonce 14,8 milliards
de paramètres, contre 2,538 milliards pour le socle de MAT-LM.

| Bras | Contrat valide | Preuves exactes | Toutes les ancres | Cible complète exacte |
|---|---:|---:|---:|---:|
| Granite 2B vierge | 0/9 | 0/9 | non mesurable | 0/9 |
| Qwen 14,8B non adapté | 4/9 | 7/9 | 3/9 | 0/9 |
| MAT-LM 2B + LoRA + calculateur | 9/9 | 9/9 | 9/9 | 9/9 |

Qwen a recopié les neuf `request_id`, n'a inventé aucune preuve et atteint un
rappel moyen de 62,0 % sur les ancres factuelles. Il paraphrase toutefois les
cibles et ne maîtrise pas systématiquement le schéma d'abstention. MAT-LM est
donc nettement meilleur **sur son protocole spécialisé** malgré un socle près
de six fois plus petit. Ce résultat ne démontre pas que MAT-LM est meilleur en
connaissances générales, rédaction ou raisonnement ouvert.

Le LoRA de 4,26 millions de paramètres est précisément le bras fine-tuné de
cette expérience : il apprend comment lire, citer, s'abstenir et demander un
calcul, tandis que les faits restent externes dans la mémoire. Le Qwen Q4_0
Ollama n'est pas directement entraînable par ce pipeline; un futur bras Qwen
LoRA exigerait un checkpoint source distinct et devra rester séparé du test
scellé.

Le protocole et les rapports Qwen sont détaillés dans
[Comparer MAT-LM à Qwen avec Ollama CLI](OLLAMA_CLI_HELDOUT_BENCHMARK.md).

Rapports publiés :

- `reports/granite-base-dev9-v8.json`, SHA-256
  `5571875152a59444009fac024c0c7a5029eb062600018014043a32b887bb46ab` ;
- `reports/matlm-v0.8-pilot-300-dev9-adapter.json`, SHA-256
  `b456ac5ea96031c13aae4d698cfda2d05f4a8b002bef4df40b509c9a2c510723` ;
- `reports/matlm-v0.8-pilot-300-dev9-verified.json`, SHA-256
  `12eb1dcfd92c5fdefe1f293419d24a578635204ccfe8951968f718022f1fa1a3`.

Le manifeste d'entraînement assaini est publié dans
`reports/matlm-v0.8-training-manifest-public.json`. Il contient les versions
logicielles, la graine, les hyperparamètres, les empreintes des données, du
tokenizer et des deux fragments de poids Granite, sans chemin de profil local.

## Séparation des données

Le curriculum v8 contient 2 250 exemples train et 270 exemples dev, répartis
également entre neuf familles. L'audit trouve zéro chevauchement
train/dev pour les mondes, identifiants, preuves, textes de preuve, réponses
cibles et objets cibles.

- train JSONL :
  `e79bd5ff4f27c572c0a6d7f122a51dd3ba9b43c7eb9f21e6f1dd56bcb76625c4` ;
- dev JSONL :
  `aa2d116ecce079e9a9e33e2df18d9bfdf8e30b3084116346fafc0a711e389a63`.

Les 2 520 exemples tiennent entièrement dans 1 024 tokens. Le maximum est
1 020 pour train et 1 016 pour dev.

## Limites et prochaine preuve

Neuf cas suffisent à valider le chemin technique, mais pas à revendiquer une
supériorité générale. Le jeu dev a influencé le débogage ; il n'est donc pas
un test officiel. Après gel du modèle et du protocole, il faut produire un
troisième split scellé d'au moins 900 cas, publier ses empreintes avant
exécution et ajouter des benchmarks publics de raisonnement et de langue.
