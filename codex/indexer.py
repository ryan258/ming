from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from engine.contracts import CapabilityCard

TOKEN_RE = re.compile(r"[A-Za-z0-9_./+-]{2,}")
_TAG_JOIN = "\x1f"


def _encode_tags(tags: tuple[str, ...]) -> str:
    return _TAG_JOIN.join(tags)


def _decode_tags(raw: str) -> tuple[str, ...]:
    if _TAG_JOIN in raw:
        return tuple(part for part in raw.split(_TAG_JOIN) if part)
    return tuple(raw.split())


class CardFormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class IndexResult:
    indexed: int
    skipped: int
    errors: tuple[str, ...]


def parse_card(path: Path, tools_root: Path) -> CapabilityCard:
    path = path.resolve()
    root = tools_root.resolve()
    if root not in path.parents:
        raise CardFormatError(f"card is outside tools root: {path}")
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise CardFormatError("missing opening YAML-style metadata delimiter")
    try:
        metadata_text, body = text[4:].split("\n---\n", 1)
    except ValueError as exc:
        raise CardFormatError("missing closing YAML-style metadata delimiter") from exc

    metadata: dict[str, str] = {}
    for raw_line in metadata_text.splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue
        if ":" not in raw_line:
            raise CardFormatError(f"invalid metadata line: {raw_line!r}")
        key, value = raw_line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("\"").strip("'")

    missing = [key for key in ("id", "title", "summary", "tags") if not metadata.get(key)]
    if missing:
        raise CardFormatError(f"missing metadata: {', '.join(missing)}")
    tags_raw = metadata["tags"].strip().strip("[]")
    tags = tuple(
        item.strip().strip("\"").strip("'")
        for item in tags_raw.split(",")
        if item.strip()
    )
    if not tags:
        raise CardFormatError("tags must contain at least one value")
    return CapabilityCard(
        card_id=metadata["id"],
        title=metadata["title"],
        summary=metadata["summary"],
        tags=tags,
        body=body.strip(),
        source=str(path.relative_to(root)),
    )


class CodexIndex:
    def __init__(self, db_path: Path, tools_root: Path) -> None:
        self.db_path = db_path
        self.tools_root = tools_root

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @staticmethod
    def fts5_available() -> bool:
        try:
            with sqlite3.connect(":memory:") as connection:
                connection.execute("CREATE VIRTUAL TABLE probe USING fts5(value)")
            return True
        except sqlite3.OperationalError:
            return False

    def initialize(self, connection: sqlite3.Connection | None = None) -> None:
        owned = connection is None
        conn = connection or self._connect()
        try:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS cards USING fts5(
                    card_id UNINDEXED,
                    title,
                    summary,
                    tags,
                    body,
                    source UNINDEXED,
                    tokenize='porter unicode61'
                )
                """
            )
            conn.commit()
        finally:
            if owned:
                conn.close()

    def rebuild(self) -> IndexResult:
        errors: list[str] = []
        cards: list[CapabilityCard] = []
        for path in sorted(self.tools_root.rglob("*.md")):
            try:
                cards.append(parse_card(path, self.tools_root))
            except (OSError, UnicodeError, CardFormatError) as exc:
                errors.append(f"{path.relative_to(self.tools_root)}: {exc}")

        with self._connect() as connection:
            self.initialize(connection)
            connection.execute("DELETE FROM cards")
            connection.executemany(
                "INSERT INTO cards(card_id, title, summary, tags, body, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        card.card_id,
                        card.title,
                        card.summary,
                        _encode_tags(card.tags),
                        card.body,
                        card.source,
                    )
                    for card in cards
                ],
            )
        return IndexResult(indexed=len(cards), skipped=len(errors), errors=tuple(errors))

    def count(self) -> int:
        if not self.db_path.exists():
            return 0
        with self._connect() as connection:
            try:
                row = connection.execute("SELECT count(*) AS count FROM cards").fetchone()
            except sqlite3.OperationalError:
                return 0
        return int(row["count"] if row else 0)

    def search(self, query: str, limit: int = 5) -> list[CapabilityCard]:
        if limit < 1 or limit > 20:
            raise ValueError("search limit must be between 1 and 20")
        if not self.db_path.exists():
            self.rebuild()
        tokens = list(dict.fromkeys(token.lower() for token in TOKEN_RE.findall(query)))[:16]
        if not tokens:
            return []
        fts_query = " OR ".join(f'"{token.replace(chr(34), "")}"' for token in tokens)
        with self._connect() as connection:
            self.initialize(connection)
            rows = connection.execute(
                """
                SELECT card_id, title, summary, tags, body, source, bm25(cards) AS rank
                FROM cards
                WHERE cards MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        return [
            CapabilityCard(
                card_id=row["card_id"],
                title=row["title"],
                summary=row["summary"],
                tags=_decode_tags(row["tags"]),
                body=row["body"],
                source=row["source"],
                score=float(row["rank"]),
            )
            for row in rows
        ]
