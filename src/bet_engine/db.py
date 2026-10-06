"""SQLite access: connection lifecycle, schema application, migrations.

Storage is deliberately boring so that provenance columns (model version, code
version, config hash) are the only interesting part of any row.
"""

# TODO: schema.sql, migration runner, typed upsert helpers.
