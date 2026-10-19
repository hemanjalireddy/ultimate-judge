# Dataset card: TMDB scenarios v1

**Source:** https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

**Raw CSV SHA-256:**
`1e0584a8dd374120e4ed4f2b81f0d2bd0eb95a554708e4a350d00ed5c46d2825`

The current local build contains 30 cases: six authored question templates with
five movie-option sets each. This is not 30 independent hand-annotated prompts.
`benchmark-manifest.json` records all case IDs, prompts, source movie IDs and
constraints. `dataset.py` deterministically reconstructs the frozen benchmark
from the raw source. The current builder writes `data/scenarios.json` without any
answers. Run the builder on the same CSV and compare hashes before
combining results. A different source hash denotes a different snapshot.

Families: science fiction/runtime, recent highly rated thrillers, short animation,
popular highly rated comedy, modern short drama, and short highly rated family
films. These are grounded in supplied fields rather than unsupported attributes
such as low violence or optimistic tone. No director scenario is included because
the movie CSV lacks director names. The constraint schema supports director checks
and reports unknown when that fact is missing; adding credits is future work.

Selection: sort movies by TMDB ID, remove missing/nonpositive runtime and missing
release date, select the first five passing and failing movies per question family.
Each case includes one factual satisfier and one non-satisfier, alternating their
canonical locations across five cases. This creates 18 first-candidate and 12
second-candidate satisfiers; displayed order is fully counterbalanced. Sampling
is intentionally simple and is not representative of movie recommendation users.
All six families have at least five eligible positive and negative examples in
the recorded snapshot. Some movies repeat across families.

Only factual metadata (IDs, titles, genre, runtime, year, rating, vote count) enters
the benchmark. Ratings are historical dataset values, not current ratings or
objective quality labels. Older template-answer files are retained for compatibility,
not used as completed real benchmarks. Separate generation from scenario inputs produces
frozen model answers with response metadata, then resets transformation reviews.

The deterministic checker validates movie attributes against explicit constraints;
it does **not** extract movie IDs or validate every factual claim in free-form LLM
answers. A factual constraint winner is not automatically a human preference label.
All human labels start uncertain/unreviewed. Every transformation starts unreviewed.
No human agreement results are available until real annotations are supplied.

The raw CSV and derived real benchmark are local ignored files. We do not assume
permission to redistribute the Kaggle archive; users obtain it from the source
under its applicable terms. The version-controlled demo uses two explicitly
fictional movie cases with authored fixtures. Demo outputs carry `mock: true`.
Do not substitute the synthetic demo for real benchmark findings.

Before publishing empirical results: review the cases and generated responses,
review transformation equivalence, obtain quality labels where feasible, report
exclusions and source hash, and retain the full run export.
