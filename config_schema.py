from __future__ import annotations

from plugins.memory.config_schema import (
    ProviderConfigSchema,
    ProviderField,
    KIND_BOOL,
    KIND_TEXT,
    STORAGE_FLAT_JSON,
)

SCHEMA = ProviderConfigSchema(
    name="living_profile",
    label="Living Profile Memory",
    storage=STORAGE_FLAT_JSON,
    fields=(
        ProviderField(
            key="auto_extract",
            label="Auto Extract Facts",
            kind=KIND_BOOL,
            default="true",
            description="Automatically extract key facts and project conventions in the background after turns.",
        ),
        ProviderField(
            key="recall_limit",
            label="Recall Limit",
            kind=KIND_TEXT,
            default="3",
            description="Maximum number of relevant profile sections / past turn snippets to inject during prefetch.",
        ),
    ),
)
CONFIG_SCHEMA = SCHEMA
