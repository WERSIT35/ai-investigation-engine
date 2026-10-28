---
doc_id: syn::postmortem-inc-1042
title: Postmortem INC-1042 — Payment processing delays (March 4–6, 2026)
type: postmortem
incident: INC-1042
date: 2026-03-08
status: final
author: Ledger team
---
# Postmortem INC-1042 — Payment processing delays

## Summary
Between 2026-03-04 08:10 and 2026-03-06 14:20 card payments were accepted by payment-api but settlement was delayed by up to 47 minutes. 12,400 transactions were affected. No payments were lost.

## Root cause
The deployment of ledger-service v4.2 on March 3 (change CHG-3301) reduced the connection pool to 15 connections per pod and switched PgBouncer to transaction mode. Under Monday morning peak load the pool was exhausted, ledger writes queued, and payment-api timed out waiting for ledger confirmations, retrying into a growing backlog.

## Affected services
- ledger-service (primary failure)
- payment-api (timeouts and retries)
- notification-service (receipt e-mails delayed because payments.settled events were late)

## Resolution
On 2026-03-06 at 14:20 the on-call engineer rolled ledger-service back to v4.1 (CHG-3318), following the runbook "Ledger service — operations runbook". The backlog drained within 35 minutes.

## Action items
1. Load-test pool settings before changing them (owner: Ledger team).
2. Alert on ledger write queue depth > 500 (owner: SRE).
