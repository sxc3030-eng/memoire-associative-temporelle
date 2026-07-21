# Memory Hub multi-IA

## But

Le Memory Hub est une mémoire locale indépendante du modèle. Qwen, Llama,
Mistral, Gemma ou un autre agent peuvent interroger les mêmes connaissances
partagées sans utiliser le même format de prompt, le même moteur d'inférence ou
les mêmes poids.

Le hub ne demande jamais à un modèle de décider seul ce qui devient vrai. Il
sépare le stockage, la récupération, le calcul déterministe et la formulation
de la réponse.

```text
question de l'IA -> adaptateur -> Memory Hub
                    IA locale <- capsule JSON bornée
```

## Séparation des espaces

La première version utilise une base `MemoryEngine` distincte par espace. Cette
séparation physique évite de confondre une similarité de contexte avec une
frontière d'accès.

| Type | Lecture | Écriture | Exemple |
|---|---|---|---|
| `private` | propriétaire seulement | propriétaire seulement | préférences d'un agent |
| `shared` | agents explicitement autorisés | auteurs explicitement autorisés | projet commun |
| `reference` | agents autorisés | aucune écriture par le hub | archives scientifiques importées hors ligne par un mainteneur |

Un simple champ `agent_id` n'est pas une authentification. Le prototype
fonctionne uniquement en processus local de confiance. Avant toute écoute
réseau, chaque agent devra utiliser une capacité secrète, et les tests devront
prouver l'isolation des espaces.

## Capsule neutre

Le modèle ne reçoit pas toute la base. Pour chaque question, le hub retourne
une capsule JSON bornée :

```json
{
  "schema_version": "memory-hub-capsule-v1",
  "agent_id": "qwen-local",
  "query_sha256": "...",
  "items": [
    {
      "space": "science-reference",
      "space_policy": "reference",
      "episode_id": "...",
      "events": [{"text": "...", "source": "observed", "context": {}}],
      "score": 4.2,
      "explanation": {}
    }
  ],
  "retrieval": {"spaces_consulted": 1, "returned": 1},
  "budget": {"character_limit": 16384, "truncated": false}
}
```

Le budget porte sur la taille sérialisée des preuves. Une capsule doit rester
déterministe pour une mémoire et une requête identiques. Les doublons sont
retirés avant de consommer le budget.

## Écriture et quatre niveaux

Les niveaux restent indépendants du modèle qui propose l'information :

1. `received` : proposition reçue d'un humain, d'un fichier ou d'une IA ;
2. `observed` : observation extérieure ou source vérifiable ;
3. `consolidated` : motif soutenu par plusieurs preuves ;
4. `operational` : procédure déterministe testée et bornée.

Une sortie de modèle est stockée au plus comme `generated` ou `inferred`. Elle
ne peut pas devenir `observed` par le seul fait d'être répétée par plusieurs
modèles. Une confirmation humaine ou une observation extérieure doit porter
sa propre provenance.

## Adaptateurs de modèles

Chaque adaptateur transforme la même capsule en un prompt approprié au modèle,
mais il ne modifie ni les preuves ni les réponses attendues. Le banc local doit
comparer, pour chaque modèle et chaque question :

- modèle seul ;
- modèle avec exactement la même capsule ;
- éventuellement modèle avec capsule et calculatrice.

Les alias d'un même modèle sont dédupliqués par empreinte. Aucun modèle n'est
téléchargé, supprimé ou remplacé par le benchmark. Les sorties publiées
contiennent le nom, l'empreinte, les paramètres annoncés, la quantification,
la latence, la réussite, l'abstention et les erreurs.

## Mesures

Le gain de mémoire est rapporté séparément de la qualité absolue du modèle :

- exactitude sans mémoire et avec mémoire ;
- différence absolue et relative ;
- réponses sans preuve et contradictions ignorées ;
- abstentions correctes lorsque la capsule est insuffisante ;
- fidélité aux sources ;
- latences p50/p95 et taille de capsule ;
- erreurs ou délais dépassés par modèle.

Le jeu de questions reste tenu à l'écart de l'import. Une réponse n'est pas
considérée correcte parce qu'elle ressemble au texte de la capsule : elle est
comparée à des fragments attendus et interdits définis séparément.

## Étapes

1. valider le constructeur de capsules et l'isolation physique ;
2. importer le corpus scientifique de référence dans une base temporaire ;
3. exécuter un test court sur chaque empreinte de modèle installée ;
4. analyser les échecs avant d'augmenter le nombre de questions ;
5. ajouter une authentification par capacités avant tout partage réseau ;
6. remplacer la consolidation globale par des mises à jour incrémentales avant
   une expérience à grande échelle.
