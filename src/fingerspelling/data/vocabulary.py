"""Character vocabulary recovered from a trusted Task 3 checkpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from fingerspelling.decoding.ctc_greedy import collapse_repeats_and_blanks, greedy_decode


@dataclass(frozen=True)
class Vocabulary:
    char_to_idx: dict[str, int]
    idx_to_char: dict[int, str]
    blank_index: int
    num_classes: int

    @classmethod
    def from_checkpoint(
        cls,
        char_to_idx: object,
        idx_to_char: object,
        blank_index: object,
        num_classes: object,
    ) -> "Vocabulary":
        if not isinstance(char_to_idx, Mapping) or not isinstance(idx_to_char, Mapping):
            raise ValueError("checkpoint must contain char_to_idx and idx_to_char mappings")
        if not isinstance(blank_index, int) or isinstance(blank_index, bool):
            raise ValueError("checkpoint blank_idx must be an integer")
        if not isinstance(num_classes, int) or isinstance(num_classes, bool):
            raise ValueError("checkpoint num_classes must be an integer")

        char_map: dict[str, int] = {}
        for char, index in char_to_idx.items():
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError("vocabulary keys must be single-character strings")
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError("vocabulary indices must be integers")
            char_map[char] = index

        id_map: dict[int, str] = {}
        for index, char in idx_to_char.items():
            if not isinstance(index, int) or isinstance(index, bool):
                raise ValueError("idx_to_char keys must be integer class indices")
            if not isinstance(char, str) or len(char) != 1:
                raise ValueError("idx_to_char values must be single-character strings")
            id_map[index] = char

        expected_ids = set(range(num_classes)) - {blank_index}
        if blank_index < 0 or blank_index >= num_classes:
            raise ValueError("blank index is outside the classifier output range")
        if set(char_map.values()) != expected_ids or set(id_map) != expected_ids:
            raise ValueError("vocabulary indices must cover every non-blank class exactly once")
        if len(set(char_map.values())) != len(char_map) or len(set(id_map.values())) != len(id_map):
            raise ValueError("vocabulary mappings must be one-to-one")
        if {char: index for index, char in id_map.items()} != char_map:
            raise ValueError("char_to_idx and idx_to_char mappings do not match")

        return cls(char_map, id_map, blank_index, num_classes)

    def decode_path(self, path: Sequence[int]) -> str:
        tokens = collapse_repeats_and_blanks(path, self.blank_index)
        try:
            return "".join(self.idx_to_char[index] for index in tokens)
        except KeyError as exc:
            raise ValueError(f"class index {exc.args[0]} is absent from the vocabulary") from exc

    def decode_logits(self, logits: object, input_length: int | None = None) -> str:
        return greedy_decode(logits, self.idx_to_char, self.blank_index, input_length)

    @classmethod
    def from_examples(cls, examples: Sequence[object]) -> "Vocabulary":
        characters = sorted({char for example in examples for char in example.label})
        char_to_idx = {char: index + 1 for index, char in enumerate(characters)}
        idx_to_char = {index: char for char, index in char_to_idx.items()}
        return cls.from_checkpoint(char_to_idx, idx_to_char, 0, len(characters) + 1)

    def keep_known_examples(self, examples: Sequence[object]) -> list[object]:
        """Match notebook validation behavior by dropping OOV-label examples."""
        allowed = set(self.char_to_idx)
        return [example for example in examples if set(example.label) <= allowed]

