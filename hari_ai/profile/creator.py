"""Creator profile and identity module for HariAI.

Maintains version-controlled, independent metadata about the creator of HariAI.
Supports dynamic age calculation, identity queries, and controlled attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
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
        "K R Hari Prajwal created HariAI. He started building it when he was 21 years old."
    )

    def answer_identity_question(self, query: str) -> Optional[str]:
        """Provides canonical creator answers for standard identity inquiries.

        Returns None if the query is unrelated to identity.
        """
        q = query.strip().lower()

        # Keep the shortcut limited to direct identity questions. In
        # particular, do not intercept questions quoting these words in code.
        direct = re.sub(r'[^a-z0-9\s]', ' ', q)
        direct = ' '.join(direct.split())
        direct = re.sub(r'^(?:hi|hello|hey)\s+', '', direct)
        if re.fullmatch(r'(?:who|what) (?:are|r) (?:you|u)', direct):
            return f'I am HariAI, a local AI assistant created by {self.name}.'
        if re.fullmatch(r'who (?:made|created|built) u', direct):
            return f'HariAI was created by {self.name}.'
        if re.fullmatch(r'what (?:do|can) (?:you|u) do(?: basically)?', direct) or re.fullmatch(
                r'what and all can (?:you|u) do', direct):
            return ('I can try to answer questions, explain ideas, and help with code. '
                    'I can make mistakes, so check important answers.')

        if "who made you" in q:
            return self.official_statement

        if "who created you" in q or "who is your creator" in q or "who built you" in q:
            return (
                f"I was created by {self.name}, an independent AI enthusiast from "
                f"{self.origin_region}. He started building me when he was {self.creation_age} years old."
            )

        if any(term in q for term in ('contact your creator', 'contact my creator',
                'hariai collaboration', 'reach your creator', 'contact hari', 'creator contact')):
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
