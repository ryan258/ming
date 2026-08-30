from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from codex.indexer import CardFormatError, CodexIndex, parse_card

VALID_CARD = """---
id: test.card
title: Test Card
summary: Reviews Python API security.
tags: [python, api, security]
---
# Method

Use bounded evidence.
"""


class CodexTests(unittest.TestCase):
    def test_parse_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "card.md"
            path.write_text(VALID_CARD)
            card = parse_card(path, root)
        self.assertEqual(card.card_id, "test.card")
        self.assertEqual(card.tags, ("python", "api", "security"))

    def test_parse_rejects_path_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "tools"
            root.mkdir()
            path = base / "outside.md"
            path.write_text(VALID_CARD)
            with self.assertRaisesRegex(CardFormatError, "outside"):
                parse_card(path, root)

    def test_rebuild_records_bad_cards_without_hiding_good_ones(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tools"
            root.mkdir()
            (root / "good.md").write_text(VALID_CARD)
            (root / "bad.md").write_text("not a card")
            index = CodexIndex(Path(tmp) / "codex.db", root)
            result = index.rebuild()
            self.assertEqual(result.indexed, 1)
            self.assertEqual(result.skipped, 1)
            self.assertEqual(index.count(), 1)

    def test_multiword_tags_round_trip(self) -> None:
        card_text = """---
id: test.card
title: Test Card
summary: Reviews image generation.
tags: [image generation, art-direction]
---
# Method

Use bounded evidence.
"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tools"
            root.mkdir()
            (root / "card.md").write_text(card_text)
            index = CodexIndex(Path(tmp) / "codex.db", root)
            index.rebuild()
            cards = index.search("image")
        self.assertEqual(cards[0].tags, ("image generation", "art-direction"))

    def test_search_returns_relevant_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tools"
            root.mkdir()
            (root / "card.md").write_text(VALID_CARD)
            index = CodexIndex(Path(tmp) / "codex.db", root)
            index.rebuild()
            cards = index.search("Python security handler")
        self.assertEqual(cards[0].card_id, "test.card")

    def test_search_handles_fts_punctuation_as_data(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "tools"
            root.mkdir()
            (root / "card.md").write_text(VALID_CARD)
            index = CodexIndex(Path(tmp) / "codex.db", root)
            index.rebuild()
            cards = index.search("api) OR NOT [secret]*")
        self.assertEqual(cards[0].card_id, "test.card")


if __name__ == "__main__":
    unittest.main()
