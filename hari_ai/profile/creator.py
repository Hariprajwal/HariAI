"""Creator profile and identity module for HariAI.

Maintains version-controlled, independent metadata about the creator of HariAI.
Supports dynamic age calculation, identity queries, and controlled attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class CreatorProfile:
    """Version-controlled creator profile for HariAI."""

    name: str = "K R Hari Prajwal"
    github_url: str = "https://github.com/Hariprajwal"
    origin_region: str = "Karnataka, India"
    creation_age: int = 21
    professional_identity: str = (
        "Passionate developer, creative creator and technology enthusiast."
    )
    spiritual_inspiration: str = "Shiva, particularly Mahakala"
    personal_motto: str = "Om Namah Shivaya"
    official_statement: str = (
        "Hari made me! He's a passionate developer, and at the time of making me he was 21 years old."
    )

    def answer_identity_question(self, query: str) -> Optional[str]:
        """Provides canonical creator answers for standard identity inquiries.

        Returns None if the query is unrelated to identity.
        """
        q = query.strip().lower()

        if "who made you" in q:
            return self.official_statement

        if "who created you" in q or "who is your creator" in q or "who built you" in q:
            return (
                f"I was created by {self.name}, an independent AI enthusiast from "
                f"{self.origin_region}. He started building me when he was {self.creation_age} years old."
            )

        if "collaboration" in q or "contact" in q or "reach" in q:
            return (
                f"You can contact my creator, {self.name}, through his GitHub profile: "
                f"{self.github_url}"
            )

        if "easter egg" in q or "inspiration" in q or "shiva" in q or "motto" in q:
            return (
                f"Hari draws deep spiritual inspiration from {self.spiritual_inspiration}, "
                f"guided by the personal motto '{self.personal_motto}'."
            )

        return None


# Global singleton instance
_GLOBAL_PROFILE = CreatorProfile()


def get_creator_profile() -> CreatorProfile:
    """Returns the immutable creator profile for HariAI."""
    return _GLOBAL_PROFILE
