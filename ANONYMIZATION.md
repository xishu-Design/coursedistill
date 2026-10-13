# Anonymization Record

The release removes:

- author names, usernames, and machine-specific paths;
- online tracking identities and credentials;
- checkpoints, weights, responses, logs, and caches;
- private or locally transformed datasets;
- paper sources and unrelated workspace projects;
- historical exploratory scripts not used by the reported method.

Public model identifiers and dataset download locations are retained for
reproduction.

The release includes one derived analysis table,
`analysis/data/future_pairs.csv`. It contains public benchmark item identifiers
and aggregate verifier outcomes only. Prompt text, generated trajectories,
checkpoints, machine paths, and tracking metadata are excluded.
