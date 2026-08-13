---
name: sqlite-schema-enforcement
description: Enforces Zero-Guessing for SQLite DDL operations and queries.
---

# SQLite Schema Enforcement Directive

1. **Zero-Guessing for SQL**: NEVER guess or hallucinate column names (e.g., `current_elo`) for SQLite queries.
2. **Mandatory Schema Audit**: Before writing any `.sql` or python script that queries the database, you MUST consult `docs/database_schema.md` using `view_file` to verify the exact table relationships and columns.
3. **No Blind Fallbacks**: If a column you expected is missing, do not attempt to guess an alternative. Halt and re-architect the query based on the ERD.
