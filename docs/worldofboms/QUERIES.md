# worldofBOMs — SPARQL Query Cookbook

All queries assume the namespaces:
- `aibom: https://aikaboom.dev/aibom#`
- `bom: bom:`

Run with: `aikaboom graph query 'PREFIX ... SELECT ... WHERE { ... }'`

## All models in the graph
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?model ?label WHERE {
    ?model a aibom:Model ;
           aibom:canonicalLabel ?label .
}
```

## Models with Apache-2.0 license
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?model ?label WHERE {
    ?model a aibom:Model ;
           aibom:canonicalLabel ?label ;
           aibom:hasVersion ?v .
    ?v aibom:hasClaim ?c .
    ?c aibom:license "Apache-2.0" .
}
```

## Claims with inter-source conflicts
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?claim ?field WHERE {
    ?ann aibom:conflictKind aibom:interSourceConflict ;
         <http://www.w3.org/1999/02/22-rdf-syntax-ns#subject> ?claim ;
         <http://www.w3.org/1999/02/22-rdf-syntax-ns#predicate> ?field .
}
```

## Models trained on a specific dataset
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?model WHERE {
    ?model aibom:trainedOn ?ds .
    ?ds aibom:canonicalLabel "rajpurkar/squad_v2" .
}
```

## Highest-trust claim per version
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?version ?canonical ?score WHERE {
    ?version aibom:canonicalClaim ?canonical .
    ?canonical aibom:trustScore ?score .
}
ORDER BY DESC(?score)
```

## Artifacts identified by an arXiv id
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?artifact WHERE {
    ?artifact aibom:identifier ?id .
    ?id aibom:platform "arxiv" ;
        aibom:value "2310.06825" .
}
```

## Claim count per artifact
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?artifact (COUNT(?c) AS ?n) WHERE {
    ?artifact aibom:hasVersion ?v .
    ?v aibom:hasClaim ?c .
}
GROUP BY ?artifact
ORDER BY DESC(?n)
```

## All votes on a specific claim
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?vote ?kind ?at WHERE {
    ?vote aibom:trustVoteFor <bom:claim/9c1d2a8f...> ;
          aibom:voteKind ?kind ;
          aibom:votedAt ?at .
}
ORDER BY DESC(?at)
```

## Potential duplicate artifacts (soft collisions)
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?a ?b WHERE {
    ?a aibom:potentialDuplicateOf ?b .
}
```

## Placeholder artifacts (unresolvable refs)
```sparql
PREFIX aibom: <https://aikaboom.dev/aibom#>
SELECT ?artifact WHERE {
    ?artifact aibom:isPlaceholder true .
}
```
