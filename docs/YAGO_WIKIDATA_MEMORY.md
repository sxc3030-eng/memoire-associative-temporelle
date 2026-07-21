# YAGO 4.5 et Wikidata comme mémoire de référence

## Verdict

YAGO 4.5 est un très bon matériau pour mettre la mémoire à rude épreuve : les
identifiants sont lisibles, la taxonomie est nettoyée, les relations sont
contraintes et les annotations temporelles sont disponibles en RDF-star. Ce
n'est toutefois pas une « vérité garantie ». Une base de connaissances peut
contenir une donnée ancienne, incomplète ou erronée; notre moteur doit donc
toujours conserver la provenance, accepter plusieurs valeurs et savoir
s'abstenir.

Point essentiel : **YAGO et Wikidata ne sont pas deux confirmations
indépendantes**. YAGO 4.5 est construit à partir de Wikidata et de Schema.org.
Le moteur leur attribue donc un seul groupe de lignée :
`wikidata-schemaorg--yago-4.5`.

Sources primaires :

- [page officielle et téléchargements YAGO 4.5](https://yago-knowledge.org/downloads/yago-4-5);
- [article YAGO 4.5, SIGIR 2024](https://doi.org/10.1145/3626772.3657876);
- [politique de licence de Wikidata](https://www.wikidata.org/wiki/Wikidata:Licensing).

La page YAGO annonce 49 millions d'entités, 109 millions de faits et une
distribution en fichiers Turtle : schéma, taxonomie, faits, faits hors
Wikipédia et métadonnées RDF-star. Elle indique aussi que YAGO applique des
contraintes SHACL. L'article explique notamment que les relations sont
filtrées par domaine et portée et que les affirmations Wikidata dites
« truthy » sont retenues. C'est un filtre de qualité utile, pas une preuve
absolue de vérité.

## Ce qui a été ajouté au prototype

`src/memory_agent/yago_import.py` importe un fichier local `.ttl`, `.nt`,
`.ntx`, `.rdf`, `.txt`, ou directement le ZIP officiel « tiny », sans réseau
et sans extraire l'archive. Le résultat va dans un SQLite séparé. Il peut donc
devenir un espace de référence en lecture seule pour plusieurs IA sans se
mélanger à leur mémoire personnelle.

Le schéma conserve :

- les entités et toutes leurs étiquettes;
- chaque triplet sujet → relation → objet;
- chaque provenance d'import (fichier, membre ZIP, ligne, version);
- les annotations RDF-star, dont les dates de début et de fin;
- toutes les valeurs concurrentes d'un même couple sujet/relation.

Un triplet qui existe uniquement à l'intérieur d'une annotation temporelle
RDF-star est matérialisé avec l'état `temporally-scoped-by-rdf-star`; il n'est
donc ni perdu, ni confondu avec une affirmation intemporelle.

Le dernier point est volontaire : deux valeurs différentes ne sont jamais
écrasées. Le rapport compte les « groupes à variantes ». Ce sont des
candidats à examiner, pas automatiquement des contradictions : une personne
peut avoir plusieurs professions ou un pays plusieurs langues officielles.

## Confiance fondée sur la provenance

Le prototype n'invente aucun pourcentage de confiance.

| Niveau | Signification |
|---|---|
| `dataset-attributed` | le triplet vient d'un dump YAGO identifié et attribué |
| `statement-attributed` | une annotation RDF-star `prov:*` attribue aussi l'affirmation elle-même |

Le champ `confidence_basis_json` conserve la lignée et porte toujours
`numeric_probability: null`. Une réponse ne devrait être élevée au rang de
« confirmée indépendamment » que si une autre source possède une autre
lignée réelle — par exemple un catalogue institutionnel ou un article
primaire, et non Wikidata réemballé par YAGO.

## Sécurité et mémoire bornée

L'importeur lit le dump en continu et écrit par lots. Il ne charge ni le ZIP,
ni le graphe entier en RAM. Avant toute lecture, il contrôle :

- la taille du fichier, le nombre de membres et les tailles décompressées;
- le ratio de compression pour refuser les bombes ZIP;
- les chemins absolus, `..`, liens symboliques et membres chiffrés;
- la taille d'une instruction Turtle, le nombre d'erreurs et le nombre total
  de triplets;
- l'UTF-8 et la structure Turtle prise en charge.

Le ZIP n'est jamais extrait, ce qui ferme aussi la voie au « zip-slip ».
L'import est idempotent et validé par petits lots : après une interruption,
relancer la même commande complète la base sans recopier les triplets déjà
présents.

Les limites par défaut acceptent le ZIP YAGO 4.5 « tiny » d'environ 200 Mo
compressés publié sur le site officiel, mais refusent volontairement le dump
complet de plusieurs gigaoctets. Une hausse des limites doit être explicite.

## Utilisation, sans API

Commencer par une inspection sans écriture :

```powershell
python scripts/import_yago.py C:\donnees\yago-4.5.0.2-tiny.zip --dry-run --strict --max-triples 100000
```

Importer ensuite dans une mémoire de référence distincte :

```powershell
python scripts/import_yago.py C:\donnees\yago-4.5.0.2-tiny.zip `
  --db C:\donnees\yago-reference.sqlite3 `
  --max-triples 1000000 `
  --report C:\donnees\yago-import-report.json
```

Pour produire un JSONL en lots que le moteur de mémoire peut reprendre hors
ligne :

```powershell
python scripts/import_yago.py C:\donnees\yago-4.5.0.2-tiny.zip `
  --db C:\donnees\yago-reference.sqlite3 `
  --memory-jsonl C:\donnees\yago-memory-records.jsonl
```

Les enregistrements exportés portent `source.type = inferred`, car un fait de
référence externe n'est pas une observation personnelle confirmée par
l'utilisateur. La base ainsi construite doit ensuite être montée comme espace
`reference` en lecture seule dans `MemoryHub`.

## Attribution et licences

La distribution YAGO 4.5 est annoncée sous **Creative Commons
Attribution-ShareAlike 3.0 (CC BY-SA 3.0)** par l'équipe YAGO de Télécom Paris.
Le lien de licence de la page officielle mène précisément à la version 3.0. Toute
redistribution d'une base dérivée doit conserver l'attribution, la licence et
vérifier les obligations de partage à l'identique. Les données structurées de
Wikidata sont publiées sous **CC0**, mais cela ne remplace pas la licence de la
distribution YAGO transformée.

Attribution recommandée dans un artefact publié :

> Ce produit utilise YAGO 4.5, équipe YAGO, Télécom Paris, sous CC BY-SA 3.0.
> YAGO 4.5 est dérivé notamment de Wikidata et de Schema.org. Référence :
> Suchanek et al., “YAGO 4.5: A Large and Clean Knowledge Base with a Rich
> Taxonomy”, SIGIR 2024, DOI 10.1145/3626772.3657876.

## Limites connues

- Le parseur couvre le sous-ensemble Turtle/Turtle-star des dumps YAGO; il ne
  prétend pas être un parseur RDF universel. Utiliser `--strict` sur un petit
  échantillon avant un long import.
- Les listes RDF, collections et certains nœuds anonymes complexes sont
  refusés en mode strict. Les faits YAGO usuels et leurs annotations RDF-star
  sont pris en charge.
- Une valeur concurrente n'est pas automatiquement une contradiction logique.
- L'importeur ne télécharge rien et ne consulte aucun endpoint SPARQL. La
  personne qui lance l'essai fournit elle-même le dump local et en vérifie la
  version.
