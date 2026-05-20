# worldofBOMs — Troubleshooting

## "Cannot import pyoxigraph"
Set `AIKABOOM_GRAPH_BACKEND=rdflib` to force the fallback. Generation is
not affected. The RDFLib backend is slower for very large graphs but
identical in behavior.

## "Cache returns a stale BOM"
The cache TTL defaults to 30 days. Bump `AIKABOOM_GRAPH_TTL_DAYS`, or run
with `--cache regen` to force a fresh generation. The new claim is added
alongside the old one — nothing is overwritten.

## "Two artifacts that should be one"
Run `aikaboom graph query 'SELECT ?a ?b WHERE { ?a <https://aikaboom.dev/aibom#potentialDuplicateOf> ?b }'`
to see soft collisions. Resolve with `aikaboom graph merge <a> <b>` after
inspecting both.

## "I want to disable the store entirely"
`AIKABOOM_GRAPH_DISABLE=1`. The system behaves exactly as before the
worldofBOMs feature.

## "My graph dir is corrupted"
Delete `~/.aikaboom/graph/` and run `aikaboom graph rebuild`. The rebuild
reconstructs the graph from `results/*.json` and replays any votes from
`~/.aikaboom/graph/votes.log`.

## "Round-trip test fails for a Golden_Set BOM"
A field in the BOM is using a predicate not yet in `vocab.py`. Add the
predicate to `vocab.py` and `SCHEMA.md`; re-run the round-trip test.

## "Resolve prompt fires when I don't want it"
Set `AIKABOOM_CACHE_POLICY_DEFAULT=use` for permanent silent caching, or
pass `--cache use` per-invocation.

## "I'm running in CI and the prompt is blocking"
Non-TTY environments degrade `--cache prompt` to `--cache use`
automatically. If you see a prompt anyway, you may have a wrapper that
fakes a TTY — pass `--cache use` explicitly.
