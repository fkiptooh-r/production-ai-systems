"""Per-tenant branding for the server-rendered pages: name, tagline, colours, examples.

Lives next to each tenant's documents in data/tenants/<tenant_id>/brand.json, so adding an
agency means adding one folder. Validated with pydantic: the colours end up inside a CSS
<style> block, where Jinja's HTML escaping doesn't protect, so they must be plain hex codes.
"""

import json
from functools import cache

from pydantic import BaseModel, Field

from rag.ingest import TENANTS_DIR

HEX_COLOUR = r"^#[0-9a-fA-F]{6}$"


class Brand(BaseModel):
    tenant_id: str
    name: str = Field(min_length=1, max_length=60)
    tagline: str = Field(max_length=120)
    accent: str = Field(pattern=HEX_COLOUR)
    accent_dark: str = Field(pattern=HEX_COLOUR)
    examples: list[str] = Field(default_factory=list, max_length=8)


@cache
def brands() -> dict[str, Brand]:
    """Every tenant with a brand.json, keyed by tenant id, in alphabetical order."""
    return {
        path.parent.name: Brand(tenant_id=path.parent.name, **json.loads(path.read_text("utf-8")))
        for path in sorted(TENANTS_DIR.glob("*/brand.json"))
    }
