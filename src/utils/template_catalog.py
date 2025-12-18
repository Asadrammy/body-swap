"""Utility helpers for template catalog management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .logger import get_logger

logger = get_logger(__name__)


@dataclass
class TemplateMetadata:
    """Dataclass describing a template entry."""

    id: str
    name: str
    category: str
    description: str
    preview_url: str
    asset_path: str
    pose_type: str
    clothing_style: str
    body_visibility: str
    recommended_subjects: List[str] = field(default_factory=list)
    background: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return serializable representation."""
        return {
            "id": self.id,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "preview_url": self.preview_url,
            "asset_path": self.asset_path,
            "pose_type": self.pose_type,
            "clothing_style": self.clothing_style,
            "body_visibility": self.body_visibility,
            "recommended_subjects": self.recommended_subjects,
            "background": self.background,
            "tags": self.tags,
        }


class TemplateCatalog:
    """Loads and caches template metadata."""

    def __init__(self, catalog_path: Optional[Path] = None):
        root = Path(__file__).parent.parent.parent
        default_path = root / "data" / "templates" / "catalog.json"
        self.catalog_path = Path(catalog_path or default_path)
        self._templates: List[TemplateMetadata] = []
        self._load_catalog()

    def _load_catalog(self) -> None:
        if not self.catalog_path.exists():
            logger.warning("Template catalog file missing at %s", self.catalog_path)
            self._templates = []
            return

        try:
            with open(self.catalog_path, "r", encoding="utf-8") as f:
                raw_entries = json.load(f)
            self._templates = [TemplateMetadata(**entry) for entry in raw_entries]
            logger.info("Loaded %s template definitions", len(self._templates))
        except Exception as exc:
            logger.error("Failed to load template catalog: %s", exc)
            self._templates = []

    def list_templates(
        self, category: Optional[str] = None, tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return templates filtered by category or tag."""
        results = self._templates
        if category and category.lower() != "all":
            results = [tpl for tpl in results if tpl.category == category]

        if tag:
            results = [
                tpl
                for tpl in results
                if tag in tpl.tags or tag == tpl.clothing_style
            ]

        return [tpl.to_dict() for tpl in results]

    def get_template(self, template_id: str) -> Optional[TemplateMetadata]:
        """Fetch template metadata by id."""
        for tpl in self._templates:
            if tpl.id == template_id:
                return tpl
        return None


