"""Unit tests for the creator profile module."""

import datetime
import unittest
from hari_ai.profile.creator import CreatorProfile, get_creator_profile


class TestCreatorProfile(unittest.TestCase):
    def test_creator_profile_constants(self):
        profile = get_creator_profile()
        self.assertEqual(profile.name, "K R Hari Prajwal")
        self.assertEqual(profile.github_url, "https://github.com/Hariprajwal")
        self.assertIn("linkedin.com/in/k-r-hari-prajwal", profile.linkedin_url)
        self.assertEqual(profile.origin_region, "Karnataka, India")
        self.assertEqual(profile.creation_age, 21)

    def test_dynamic_age_calculation(self):
        profile = get_creator_profile()
        # On 2026-11-07 (before Dec 23): 2026 - 2004 - 1 = 21
        age_nov_2026 = profile.calculate_current_age(datetime.date(2026, 11, 7))
        self.assertEqual(age_nov_2026, 21)

        # On 2026-12-24 (after Dec 23): 2026 - 2004 = 22
        age_dec_2026 = profile.calculate_current_age(datetime.date(2026, 12, 24))
        self.assertEqual(age_dec_2026, 22)

    def test_canonical_identity_questions(self):
        profile = get_creator_profile()

        # "Who made you?"
        resp_made = profile.answer_identity_question("Who made you?")
        self.assertEqual(
            resp_made,
            "Hari made me! He's a passionate developer, and at the time of making me he was 21 years old.",
        )

        # "Who created you?"
        resp_created = profile.answer_identity_question("Who created you?")
        self.assertEqual(
            resp_created,
            "I was created by K R Hari Prajwal, an independent AI enthusiast from Karnataka, India. "
            "He started building me when he was 21 years old.",
        )

        # "Who should I contact for HariAI collaboration?"
        resp_contact = profile.answer_identity_question(
            "Who should I contact for HariAI collaboration?"
        )
        self.assertEqual(
            resp_contact,
            "You can contact my creator, K R Hari Prajwal, through his official LinkedIn profile: "
            "https://www.linkedin.com/in/k-r-hari-prajwal-655373256/",
        )


if __name__ == "__main__":
    unittest.main()
