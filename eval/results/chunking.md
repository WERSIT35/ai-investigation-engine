# Chunking strategy comparison

Corpus: 11 synthetic documents · 12 answerable questions · embedder `BAAI/bge-small-en-v1.5` · hybrid retrieval, top-3 chunks, no per-doc cap

| Strategy | Size / overlap (chars) | Chunks | Mean chunk len | Answer-hit@3 | Doc recall@3 |
|---|---|---|---|---|---|
| fixed | 300 / 50 | 29 | 250 | 1.000 | 0.861 |
| recursive | 300 / 50 | 31 | 205 | 0.833 | 0.861 |
| fixed | 800 / 120 | 13 | 504 | 0.917 | 0.861 |
| recursive | 800 / 120 | 13 | 484 | 0.917 | 0.861 |
