---
doc_id: syn::changelog-2026-03
title: Production change log — March 2026
type: change-log
date: 2026-03
---
# Production change log — March 2026

## CHG-3301 — 2026-03-03 16:40 — ledger-service v4.2
Deployed ledger-service v4.2. Main change: the database connection pool was switched from session mode to PgBouncer transaction mode and the pool size per pod was reduced from 40 to 15 to cut idle connections on the primary Postgres cluster. Approved by the Ledger team. Risk: medium. Rollback plan: redeploy v4.1.

## CHG-3302 — 2026-03-05 10:15 — api-gateway rate limiter
Raised the per-merchant rate limit on api-gateway from 200 to 300 requests per second for the spring campaign. Risk: low.

## CHG-3310 — 2026-03-10 09:00 — Kafka partitions
Increased partitions of the topic payments.settled from 12 to 24 to prepare for higher volume. Consumers were not redeployed. Risk: low.

## CHG-3318 — 2026-03-06 14:05 — ledger-service rollback
Rolled back ledger-service from v4.2 to v4.1 as part of incident INC-1042. Pool configuration restored to session mode, 40 connections per pod.

## CHG-3325 — 2026-03-20 11:30 — notification-service consumer fix
Redeployed notification-service with consumer group rebalancing enabled so that it reads all 24 partitions of payments.settled. Linked to INC-1057.
