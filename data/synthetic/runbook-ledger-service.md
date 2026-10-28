---
doc_id: syn::runbook-ledger-service
title: Ledger service — operations runbook
type: runbook
service: ledger-service
date: 2026-02-14
---
# Ledger service — operations runbook

Owner: Ledger team. Escalation: SRE on-call.

## Symptoms of connection pool exhaustion
- ledger write latency p95 above 2 seconds
- "remaining connection slots are reserved" errors in ledger-service logs
- growing queue of pending settlement writes

## Recovery procedure
1. Confirm the current version with `deployctl status ledger-service`.
2. If the incident started after a deployment, roll back: `deployctl rollback ledger-service --to <previous-version>`.
3. Restore the pool settings: session mode, 40 connections per pod.
4. Watch the settlement backlog until it drains, then notify payment-api owners.

## Dependencies
ledger-service writes to the primary Postgres cluster and publishes payments.settled events to Kafka.
