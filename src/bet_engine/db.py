"""SQLite access: connection lifecycle, schema application, migrations.

Storage is deliberately boring so that provenance columns (model version, code
version, config hash) are the only interesting part of any row.
"""

# TODO: DDL schema, migration runner, typed upsert helpers.
