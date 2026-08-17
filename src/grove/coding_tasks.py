from __future__ import annotations

from dataclasses import dataclass

from grove.models import DatasetRole, Task
from grove.verifiers import PythonCase, PythonSuite


@dataclass(frozen=True, slots=True)
class CodingTask:
    task: Task
    suite: PythonSuite
    reference_solution: str
    role: DatasetRole


def _task(
    task_id: str,
    prompt: str,
    cases: list[tuple[object, object]],
    solution: str,
    role: DatasetRole,
    *,
    family: str,
) -> CodingTask:
    task = Task(
        id=task_id,
        prompt=prompt,
        verifier="sandboxed_python",
        cohort=role.value,
        tags=("python", family),
        metadata={"failure_type": family, "language": "python"},
    )
    version = {
        "escaped_path": "escaped-path-v2",
        "path_restructure": "path-restructure-v1",
    }.get(family, "python-core-v1")
    return CodingTask(
        task=task,
        suite=PythonSuite(
            task_id,
            tuple(PythonCase(payload, expected) for payload, expected in cases),
            version=version,
        ),
        reference_solution=solution.strip(),
        role=role,
    )


SPLIT_PATH = r"""
def _split(path):
    parts, current, escaped = [], [], False
    for char in path:
        if escaped:
            current.append(char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == ".":
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    if escaped:
        current.append("\\")
    parts.append("".join(current))
    return parts
"""

# EXP-002 replay cohort: short single-function core-Python tasks authored blind
# to the escaped-path training family.  Each entry is
# (task_id, prompt, [(payload, expected), ...], reference_solution).
_CORE_REGRESSION_SPECS: tuple[
    tuple[str, str, list[tuple[object, object]], str], ...
] = (
    (
        "reg_sum_all",
        "Define solve(payload). payload is a list of integers. Return the sum of all of them.",
        [([1, 2, 3], 6), ([], 0), ([-5, 5, 2], 2)],
        "def solve(payload):\n    return sum(payload)",
    ),
    (
        "reg_max_value",
        "Define solve(payload). payload is a non-empty list of integers. Return the largest one.",
        [([3, 1, 2], 3), ([-7, -2], -2), ([5], 5)],
        "def solve(payload):\n    return max(payload)",
    ),
    (
        "reg_min_value",
        "Define solve(payload). payload is a non-empty list of integers. Return the smallest one.",
        [([3, 1, 2], 1), ([-7, -2], -7), ([5], 5)],
        "def solve(payload):\n    return min(payload)",
    ),
    (
        "reg_count_items",
        "Define solve(payload). payload is a list. Return how many items it contains.",
        [([1, 2, 3], 3), ([], 0), (["a", "b"], 2)],
        "def solve(payload):\n    return len(payload)",
    ),
    (
        "reg_reverse_list",
        "Define solve(payload). payload is a list. Return a new list with the items in reverse order.",
        [([1, 2, 3], [3, 2, 1]), ([], []), (["a"], ["a"])],
        "def solve(payload):\n    return payload[::-1]",
    ),
    (
        "reg_sort_ascending",
        "Define solve(payload). payload is a list of integers. Return a new list sorted from smallest to largest.",
        [([3, 1, 2], [1, 2, 3]), ([], []), ([2, 2, 1], [1, 2, 2])],
        "def solve(payload):\n    return sorted(payload)",
    ),
    (
        "reg_sort_descending",
        "Define solve(payload). payload is a list of integers. Return a new list sorted from largest to smallest.",
        [([3, 1, 2], [3, 2, 1]), ([], []), ([1, 5, 3], [5, 3, 1])],
        "def solve(payload):\n    return sorted(payload, reverse=True)",
    ),
    (
        "reg_sum_odd",
        "Define solve(payload). payload is a list of integers. Return the sum of only the odd integers.",
        [([1, 2, 3, 4], 4), ([], 0), ([2, 4], 0)],
        "def solve(payload):\n    return sum(value for value in payload if value % 2 != 0)",
    ),
    (
        "reg_count_even",
        "Define solve(payload). payload is a list of integers. Return how many of them are even.",
        [([1, 2, 3, 4], 2), ([], 0), ([2, 4, 6], 3)],
        "def solve(payload):\n    return sum(1 for value in payload if value % 2 == 0)",
    ),
    (
        "reg_count_odd",
        "Define solve(payload). payload is a list of integers. Return how many of them are odd.",
        [([1, 2, 3, 4], 2), ([], 0), ([2, 4], 0)],
        "def solve(payload):\n    return sum(1 for value in payload if value % 2 != 0)",
    ),
    (
        "reg_double_each",
        "Define solve(payload). payload is a list of integers. Return a new list where each integer is multiplied by two.",
        [([1, 2], [2, 4]), ([], []), ([-3], [-6])],
        "def solve(payload):\n    return [value * 2 for value in payload]",
    ),
    (
        "reg_square_each",
        "Define solve(payload). payload is a list of integers. Return a new list where each integer is squared.",
        [([1, 2, 3], [1, 4, 9]), ([], []), ([-2], [4])],
        "def solve(payload):\n    return [value * value for value in payload]",
    ),
    (
        "reg_negate_each",
        "Define solve(payload). payload is a list of integers. Return a new list where the sign of each integer is flipped.",
        [([1, -2], [-1, 2]), ([], []), ([0], [0])],
        "def solve(payload):\n    return [-value for value in payload]",
    ),
    (
        "reg_abs_each",
        "Define solve(payload). payload is a list of integers. Return a new list holding the absolute value of each integer.",
        [([-1, 2, -3], [1, 2, 3]), ([], []), ([0], [0])],
        "def solve(payload):\n    return [abs(value) for value in payload]",
    ),
    (
        "reg_increment_each",
        "Define solve(payload). payload is a list of integers. Return a new list where each integer is increased by one.",
        [([1, 2], [2, 3]), ([], []), ([-1], [0])],
        "def solve(payload):\n    return [value + 1 for value in payload]",
    ),
    (
        "reg_filter_positive",
        "Define solve(payload). payload is a list of integers. Return a new list keeping only integers greater than zero, in order.",
        [([1, -2, 3], [1, 3]), ([], []), ([-1, 0], [])],
        "def solve(payload):\n    return [value for value in payload if value > 0]",
    ),
    (
        "reg_filter_negative",
        "Define solve(payload). payload is a list of integers. Return a new list keeping only integers less than zero, in order.",
        [([1, -2, 3], [-2]), ([], []), ([-1, -4], [-1, -4])],
        "def solve(payload):\n    return [value for value in payload if value < 0]",
    ),
    (
        "reg_filter_nonzero",
        "Define solve(payload). payload is a list of integers. Return a new list with every zero removed, keeping order.",
        [([0, 1, 0, 2], [1, 2]), ([], []), ([0, 0], [])],
        "def solve(payload):\n    return [value for value in payload if value != 0]",
    ),
    (
        "reg_first_item",
        "Define solve(payload). payload is a non-empty list. Return its first item.",
        [([9, 2], 9), (["a", "b"], "a"), ([5], 5)],
        "def solve(payload):\n    return payload[0]",
    ),
    (
        "reg_last_item",
        "Define solve(payload). payload is a non-empty list. Return its last item.",
        [([9, 2], 2), (["a", "b"], "b"), ([5], 5)],
        "def solve(payload):\n    return payload[-1]",
    ),
    (
        "reg_sum_first_last",
        "Define solve(payload). payload is a non-empty list of integers. Return the sum of the first and last items. With one item, add it to itself.",
        [([1, 2, 3], 4), ([5], 10), ([2, 7], 9)],
        "def solve(payload):\n    return payload[0] + payload[-1]",
    ),
    (
        "reg_range_span",
        "Define solve(payload). payload is a non-empty list of integers. Return the largest value minus the smallest value.",
        [([3, 1, 9], 8), ([5], 0), ([-2, 4], 6)],
        "def solve(payload):\n    return max(payload) - min(payload)",
    ),
    (
        "reg_average_floor",
        "Define solve(payload). payload is a non-empty list of integers. Return the floor of the mean, computed with integer floor division of the sum by the count.",
        [([1, 2, 3], 2), ([1, 2], 1), ([-3, -4], -4)],
        "def solve(payload):\n    return sum(payload) // len(payload)",
    ),
    (
        "reg_count_positive",
        "Define solve(payload). payload is a list of integers. Return how many are greater than zero.",
        [([1, -2, 3], 2), ([], 0), ([0], 0)],
        "def solve(payload):\n    return sum(1 for value in payload if value > 0)",
    ),
    (
        "reg_count_negative",
        "Define solve(payload). payload is a list of integers. Return how many are less than zero.",
        [([1, -2, 3], 1), ([], 0), ([-1, -1], 2)],
        "def solve(payload):\n    return sum(1 for value in payload if value < 0)",
    ),
    (
        "reg_product_all",
        "Define solve(payload). payload is a list of integers. Return the product of all of them; return 1 for an empty list.",
        [([2, 3, 4], 24), ([], 1), ([5, -1], -5)],
        "def solve(payload):\n    result = 1\n    for value in payload:\n        result *= value\n    return result",
    ),
    (
        "reg_unique_count",
        "Define solve(payload). payload is a list of JSON scalar values. Return how many distinct values it contains.",
        [([1, 1, 2], 2), ([], 0), (["a", "b", "a"], 2)],
        "def solve(payload):\n    return len(set(payload))",
    ),
    (
        "reg_contains_zero",
        "Define solve(payload). payload is a list of integers. Return true when the list contains at least one zero, otherwise false.",
        [([1, 0], True), ([], False), ([3, 4], False)],
        "def solve(payload):\n    return 0 in payload",
    ),
    (
        "reg_all_positive",
        "Define solve(payload). payload is a list of integers. Return true when every integer is greater than zero; an empty list counts as true.",
        [([1, 2], True), ([], True), ([1, 0], False)],
        "def solve(payload):\n    return all(value > 0 for value in payload)",
    ),
    (
        "reg_any_negative",
        "Define solve(payload). payload is a list of integers. Return true when at least one integer is less than zero, otherwise false.",
        [([1, -1], True), ([], False), ([0, 2], False)],
        "def solve(payload):\n    return any(value < 0 for value in payload)",
    ),
    (
        "reg_second_largest",
        "Define solve(payload). payload is a list of integers containing at least two distinct values. Return the second largest distinct value.",
        [([1, 3, 2], 2), ([5, 5, 4], 4), ([-1, -2, -3], -2)],
        "def solve(payload):\n    return sorted(set(payload))[-2]",
    ),
    (
        "reg_index_of_max",
        "Define solve(payload). payload is a non-empty list of integers. Return the index of the first occurrence of the largest value.",
        [([1, 3, 2], 1), ([7], 0), ([2, 9, 9], 1)],
        "def solve(payload):\n    return payload.index(max(payload))",
    ),
    (
        "reg_clamp_values",
        "Define solve(payload). payload is a list of integers. Return a new list where each integer is clamped into the inclusive range 0 to 10.",
        [([-5, 3, 20], [0, 3, 10]), ([], []), ([10, 0], [10, 0])],
        "def solve(payload):\n    return [min(10, max(0, value)) for value in payload]",
    ),
    (
        "reg_evens_only",
        "Define solve(payload). payload is a list of integers. Return a new list keeping only the even integers, in order.",
        [([1, 2, 3, 4], [2, 4]), ([], []), ([1, 3], [])],
        "def solve(payload):\n    return [value for value in payload if value % 2 == 0]",
    ),
    (
        "reg_odds_only",
        "Define solve(payload). payload is a list of integers. Return a new list keeping only the odd integers, in order.",
        [([1, 2, 3], [1, 3]), ([], []), ([2, 4], [])],
        "def solve(payload):\n    return [value for value in payload if value % 2 != 0]",
    ),
    (
        "reg_running_total",
        "Define solve(payload). payload is a list of integers. Return the list of running totals, where position i holds the sum of the first i+1 integers.",
        [([1, 2, 3], [1, 3, 6]), ([], []), ([5, -5], [5, 0])],
        "def solve(payload):\n    totals, current = [], 0\n    for value in payload:\n        current += value\n        totals.append(current)\n    return totals",
    ),
    (
        "reg_adjacent_diffs",
        "Define solve(payload). payload is a list of integers. Return a list of the differences between each item and the one before it, so the result has one fewer item.",
        [([1, 4, 9], [3, 5]), ([5], []), ([2, 2, 2], [0, 0])],
        "def solve(payload):\n    return [payload[i] - payload[i - 1] for i in range(1, len(payload))]",
    ),
    (
        "reg_repeat_each",
        "Define solve(payload). payload is a list. Return a new list where every item appears twice in a row, preserving order.",
        [([1, 2], [1, 1, 2, 2]), ([], []), (["a"], ["a", "a"])],
        "def solve(payload):\n    result = []\n    for value in payload:\n        result.append(value)\n        result.append(value)\n    return result",
    ),
    (
        "reg_drop_first",
        "Define solve(payload). payload is a list. Return a new list without the first item; an empty list stays empty.",
        [([1, 2, 3], [2, 3]), ([7], []), ([], [])],
        "def solve(payload):\n    return payload[1:]",
    ),
    (
        "reg_drop_last",
        "Define solve(payload). payload is a list. Return a new list without the last item; an empty list stays empty.",
        [([1, 2, 3], [1, 2]), ([7], []), ([], [])],
        "def solve(payload):\n    return payload[:-1]",
    ),
    (
        "reg_first_two",
        "Define solve(payload). payload is a list. Return a new list holding at most its first two items.",
        [([1, 2, 3], [1, 2]), ([9], [9]), ([], [])],
        "def solve(payload):\n    return payload[:2]",
    ),
    (
        "reg_concat_lists",
        "Define solve(payload). payload is an object with list fields a and b. Return a single list holding the items of a followed by the items of b.",
        [
            ({"a": [1], "b": [2, 3]}, [1, 2, 3]),
            ({"a": [], "b": []}, []),
            ({"a": ["x"], "b": []}, ["x"]),
        ],
        'def solve(payload):\n    return payload["a"] + payload["b"]',
    ),
    (
        "reg_zip_sum",
        "Define solve(payload). payload is an object with integer list fields a and b of equal length. Return a list where each position holds the sum of the matching items.",
        [
            ({"a": [1, 2], "b": [3, 4]}, [4, 6]),
            ({"a": [], "b": []}, []),
            ({"a": [0], "b": [-1]}, [-1]),
        ],
        'def solve(payload):\n    return [x + y for x, y in zip(payload["a"], payload["b"])]',
    ),
    (
        "reg_lists_equal",
        "Define solve(payload). payload is an object with list fields a and b. Return true when the two lists hold the same items in the same order.",
        [
            ({"a": [1], "b": [1]}, True),
            ({"a": [1], "b": [2]}, False),
            ({"a": [], "b": []}, True),
        ],
        'def solve(payload):\n    return payload["a"] == payload["b"]',
    ),
    (
        "reg_common_values",
        "Define solve(payload). payload is an object with integer list fields a and b. Return the sorted list of distinct integers that appear in both.",
        [
            ({"a": [1, 2, 3], "b": [2, 3, 4]}, [2, 3]),
            ({"a": [], "b": [1]}, []),
            ({"a": [5, 5], "b": [5]}, [5]),
        ],
        'def solve(payload):\n    return sorted(set(payload["a"]) & set(payload["b"]))',
    ),
    (
        "reg_only_in_first",
        "Define solve(payload). payload is an object with integer list fields a and b. Return the sorted list of distinct integers that appear in a but not in b.",
        [
            ({"a": [1, 2, 3], "b": [2]}, [1, 3]),
            ({"a": [], "b": [1]}, []),
            ({"a": [4, 4, 5], "b": [5]}, [4]),
        ],
        'def solve(payload):\n    return sorted(set(payload["a"]) - set(payload["b"]))',
    ),
    (
        "reg_uppercase",
        "Define solve(payload). payload is a string. Return it converted to upper case.",
        [("abc", "ABC"), ("", ""), ("MiXed", "MIXED")],
        "def solve(payload):\n    return payload.upper()",
    ),
    (
        "reg_lowercase",
        "Define solve(payload). payload is a string. Return it converted to lower case.",
        [("ABC", "abc"), ("", ""), ("MiXed", "mixed")],
        "def solve(payload):\n    return payload.lower()",
    ),
    (
        "reg_strip_spaces",
        "Define solve(payload). payload is a string. Return it with leading and trailing whitespace removed.",
        [("  hi  ", "hi"), ("hi", "hi"), ("   ", "")],
        "def solve(payload):\n    return payload.strip()",
    ),
    (
        "reg_reverse_string",
        "Define solve(payload). payload is a string. Return the string with its characters in reverse order.",
        [("abc", "cba"), ("", ""), ("ab", "ba")],
        "def solve(payload):\n    return payload[::-1]",
    ),
    (
        "reg_string_length",
        "Define solve(payload). payload is a string. Return the number of characters it contains.",
        [("abc", 3), ("", 0), ("a b", 3)],
        "def solve(payload):\n    return len(payload)",
    ),
    (
        "reg_count_vowels",
        "Define solve(payload). payload is a string. Return how many characters are vowels a, e, i, o, or u, ignoring case.",
        [("apple", 2), ("", 0), ("AEx", 2)],
        'def solve(payload):\n    return sum(1 for char in payload.lower() if char in "aeiou")',
    ),
    (
        "reg_count_char",
        "Define solve(payload). payload is an object with string fields text and char, where char is one character. Return how many times char occurs in text.",
        [
            ({"text": "banana", "char": "a"}, 3),
            ({"text": "", "char": "z"}, 0),
            ({"text": "zz", "char": "z"}, 2),
        ],
        'def solve(payload):\n    return payload["text"].count(payload["char"])',
    ),
    (
        "reg_first_word",
        "Define solve(payload). payload is a string containing at least one whitespace-separated word. Return the first word.",
        [("hello world", "hello"), ("one", "one"), ("  padded start", "padded")],
        "def solve(payload):\n    return payload.split()[0]",
    ),
    (
        "reg_word_count",
        "Define solve(payload). payload is a string. Return the number of whitespace-separated words it contains.",
        [("a b c", 3), ("", 0), ("  spaced  out ", 2)],
        "def solve(payload):\n    return len(payload.split())",
    ),
    (
        "reg_capitalize_words",
        "Define solve(payload). payload is a string of whitespace-separated words. Return the words each capitalized (first letter upper, rest lower), joined by single spaces.",
        [("hello world", "Hello World"), ("a", "A"), ("", "")],
        'def solve(payload):\n    return " ".join(word.capitalize() for word in payload.split())',
    ),
    (
        "reg_replace_spaces",
        "Define solve(payload). payload is a string. Return it with every space character replaced by an underscore.",
        [("a b c", "a_b_c"), ("abc", "abc"), (" ", "_")],
        'def solve(payload):\n    return payload.replace(" ", "_")',
    ),
    (
        "reg_remove_digits",
        "Define solve(payload). payload is a string. Return it with every digit character removed.",
        [("a1b2", "ab"), ("123", ""), ("abc", "abc")],
        'def solve(payload):\n    return "".join(char for char in payload if not char.isdigit())',
    ),
    (
        "reg_keep_digits",
        "Define solve(payload). payload is a string. Return a string holding only its digit characters, in order.",
        [("a1b2", "12"), ("abc", ""), ("007", "007")],
        'def solve(payload):\n    return "".join(char for char in payload if char.isdigit())',
    ),
    (
        "reg_starts_with",
        "Define solve(payload). payload is an object with string fields text and prefix. Return true when text starts with prefix.",
        [
            ({"text": "hello", "prefix": "he"}, True),
            ({"text": "hello", "prefix": "lo"}, False),
            ({"text": "", "prefix": ""}, True),
        ],
        'def solve(payload):\n    return payload["text"].startswith(payload["prefix"])',
    ),
    (
        "reg_ends_with",
        "Define solve(payload). payload is an object with string fields text and suffix. Return true when text ends with suffix.",
        [
            ({"text": "hello", "suffix": "lo"}, True),
            ({"text": "hello", "suffix": "he"}, False),
            ({"text": "", "suffix": ""}, True),
        ],
        'def solve(payload):\n    return payload["text"].endswith(payload["suffix"])',
    ),
    (
        "reg_contains_sub",
        "Define solve(payload). payload is an object with string fields text and sub. Return true when sub occurs anywhere inside text.",
        [
            ({"text": "hello", "sub": "ell"}, True),
            ({"text": "hello", "sub": "z"}, False),
            ({"text": "", "sub": ""}, True),
        ],
        'def solve(payload):\n    return payload["sub"] in payload["text"]',
    ),
    (
        "reg_repeat_text",
        "Define solve(payload). payload is an object with string field text and non-negative integer field times. Return text repeated times times.",
        [
            ({"text": "ab", "times": 3}, "ababab"),
            ({"text": "ab", "times": 0}, ""),
            ({"text": "x", "times": 1}, "x"),
        ],
        'def solve(payload):\n    return payload["text"] * payload["times"]',
    ),
    (
        "reg_join_hyphen",
        "Define solve(payload). payload is a list of strings. Return them joined into one string with a hyphen between neighbours.",
        [(["a", "b"], "a-b"), ([], ""), (["x"], "x")],
        'def solve(payload):\n    return "-".join(payload)',
    ),
    (
        "reg_split_commas",
        "Define solve(payload). payload is a string. Return the list produced by splitting it on every comma; empty pieces are kept.",
        [("a,b,c", ["a", "b", "c"]), ("a", ["a"]), (("a,,b"), ["a", "", "b"])],
        'def solve(payload):\n    return payload.split(",")',
    ),
    (
        "reg_char_list",
        "Define solve(payload). payload is a string. Return a list of its characters in order, one string per character.",
        [("abc", ["a", "b", "c"]), ("", []), ("hi", ["h", "i"])],
        "def solve(payload):\n    return list(payload)",
    ),
    (
        "reg_swap_case",
        "Define solve(payload). payload is a string. Return it with upper-case letters made lower case and lower-case letters made upper case.",
        [("aBc", "AbC"), ("", ""), ("XY", "xy")],
        "def solve(payload):\n    return payload.swapcase()",
    ),
    (
        "reg_is_palindrome",
        "Define solve(payload). payload is a string. Return true when it reads the same forwards and backwards, comparing characters exactly.",
        [("aba", True), ("ab", False), ("", True)],
        "def solve(payload):\n    return payload == payload[::-1]",
    ),
    (
        "reg_initials",
        "Define solve(payload). payload is a string containing at least one whitespace-separated word. Return the first character of each word concatenated in order.",
        [("ada lovelace", "al"), ("x", "x"), ("big Bad Wolf", "bBW")],
        'def solve(payload):\n    return "".join(word[0] for word in payload.split())',
    ),
    (
        "reg_longest_word",
        "Define solve(payload). payload is a string containing at least one whitespace-separated word. Return the longest word; on a tie return the earliest.",
        [("a bb cc", "bb"), ("one three two", "three"), ("solo", "solo")],
        "def solve(payload):\n    return max(payload.split(), key=len)",
    ),
    (
        "reg_shortest_word",
        "Define solve(payload). payload is a string containing at least one whitespace-separated word. Return the shortest word; on a tie return the earliest.",
        [("three to xy", "to"), ("solo", "solo"), ("bb a cc", "a")],
        "def solve(payload):\n    return min(payload.split(), key=len)",
    ),
    (
        "reg_count_spaces",
        "Define solve(payload). payload is a string. Return how many space characters it contains.",
        [("a b c", 2), ("abc", 0), ("  ", 2)],
        'def solve(payload):\n    return payload.count(" ")',
    ),
    (
        "reg_truncate_five",
        "Define solve(payload). payload is a string. Return at most its first five characters.",
        [("abcdefgh", "abcde"), ("ab", "ab"), ("", "")],
        "def solve(payload):\n    return payload[:5]",
    ),
    (
        "reg_dict_keys_sorted",
        "Define solve(payload). payload is an object with string keys. Return the list of its keys sorted alphabetically.",
        [({"b": 1, "a": 2}, ["a", "b"]), ({}, []), ({"x": 0}, ["x"])],
        "def solve(payload):\n    return sorted(payload)",
    ),
    (
        "reg_dict_values_sum",
        "Define solve(payload). payload is an object whose values are integers. Return the sum of the values.",
        [({"a": 1, "b": 2}, 3), ({}, 0), ({"x": -5}, -5)],
        "def solve(payload):\n    return sum(payload.values())",
    ),
    (
        "reg_dict_get",
        "Define solve(payload). payload has an object field data and a string field key. Return the value stored under key in data, or null when the key is absent.",
        [
            ({"data": {"a": 1}, "key": "a"}, 1),
            ({"data": {}, "key": "a"}, None),
            ({"data": {"a": None}, "key": "b"}, None),
        ],
        'def solve(payload):\n    return payload["data"].get(payload["key"])',
    ),
    (
        "reg_dict_has_key",
        "Define solve(payload). payload has an object field data and a string field key. Return true when key is present in data.",
        [
            ({"data": {"a": 1}, "key": "a"}, True),
            ({"data": {"a": 1}, "key": "b"}, False),
            ({"data": {}, "key": "x"}, False),
        ],
        'def solve(payload):\n    return payload["key"] in payload["data"]',
    ),
    (
        "reg_dict_size",
        "Define solve(payload). payload is an object. Return how many key/value pairs it contains.",
        [({"a": 1, "b": 2}, 2), ({}, 0), ({"x": None}, 1)],
        "def solve(payload):\n    return len(payload)",
    ),
    (
        "reg_dict_invert",
        "Define solve(payload). payload is an object whose values are distinct strings. Return a new object mapping each value back to its key.",
        [
            ({"a": "x"}, {"x": "a"}),
            ({}, {}),
            ({"a": "1", "b": "2"}, {"1": "a", "2": "b"}),
        ],
        "def solve(payload):\n    return {value: key for key, value in payload.items()}",
    ),
    (
        "reg_dict_max_key",
        "Define solve(payload). payload is a non-empty object whose values are integers with a unique maximum. Return the key holding the largest value.",
        [
            ({"a": 1, "b": 3}, "b"),
            ({"x": 5}, "x"),
            ({"a": -1, "b": -2}, "a"),
        ],
        "def solve(payload):\n    return max(payload, key=lambda key: payload[key])",
    ),
    (
        "reg_dict_positive_only",
        "Define solve(payload). payload is an object whose values are integers. Return a new object keeping only the entries whose value is greater than zero.",
        [
            ({"a": 1, "b": -1}, {"a": 1}),
            ({}, {}),
            ({"a": 0}, {}),
        ],
        "def solve(payload):\n    return {key: value for key, value in payload.items() if value > 0}",
    ),
    (
        "reg_dict_increment",
        "Define solve(payload). payload is an object whose values are integers. Return a new object with the same keys and every value increased by one.",
        [
            ({"a": 1}, {"a": 2}),
            ({}, {}),
            ({"x": -1, "y": 0}, {"x": 0, "y": 1}),
        ],
        "def solve(payload):\n    return {key: value + 1 for key, value in payload.items()}",
    ),
    (
        "reg_merge_dicts",
        "Define solve(payload). payload has object fields a and b. Return a new object with every entry of a and b; when a key appears in both, b's value wins.",
        [
            ({"a": {"x": 1}, "b": {"y": 2}}, {"x": 1, "y": 2}),
            ({"a": {"x": 1}, "b": {"x": 9}}, {"x": 9}),
            ({"a": {}, "b": {}}, {}),
        ],
        'def solve(payload):\n    return {**payload["a"], **payload["b"]}',
    ),
    (
        "reg_word_frequency",
        "Define solve(payload). payload is a string. Return an object mapping each whitespace-separated word to how many times it occurs.",
        [
            ("a b a", {"a": 2, "b": 1}),
            ("", {}),
            ("z", {"z": 1}),
        ],
        "def solve(payload):\n    counts = {}\n    for word in payload.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts",
    ),
    (
        "reg_char_frequency",
        "Define solve(payload). payload is a string. Return an object mapping each character to how many times it occurs.",
        [
            ("aab", {"a": 2, "b": 1}),
            ("", {}),
            ("xyx", {"x": 2, "y": 1}),
        ],
        "def solve(payload):\n    counts = {}\n    for char in payload:\n        counts[char] = counts.get(char, 0) + 1\n    return counts",
    ),
    (
        "reg_add_two",
        "Define solve(payload). payload has integer fields a and b. Return their sum.",
        [({"a": 2, "b": 3}, 5), ({"a": -1, "b": 1}, 0), ({"a": 0, "b": 0}, 0)],
        'def solve(payload):\n    return payload["a"] + payload["b"]',
    ),
    (
        "reg_multiply_two",
        "Define solve(payload). payload has integer fields a and b. Return their product.",
        [({"a": 2, "b": 3}, 6), ({"a": -2, "b": 4}, -8), ({"a": 0, "b": 9}, 0)],
        'def solve(payload):\n    return payload["a"] * payload["b"]',
    ),
    (
        "reg_subtract_two",
        "Define solve(payload). payload has integer fields a and b. Return a minus b.",
        [({"a": 5, "b": 3}, 2), ({"a": 3, "b": 5}, -2), ({"a": 0, "b": 0}, 0)],
        'def solve(payload):\n    return payload["a"] - payload["b"]',
    ),
    (
        "reg_floor_divide",
        "Define solve(payload). payload has integer fields a and b, with b never zero. Return the floor division of a by b.",
        [({"a": 7, "b": 2}, 3), ({"a": -7, "b": 2}, -4), ({"a": 9, "b": 3}, 3)],
        'def solve(payload):\n    return payload["a"] // payload["b"]',
    ),
    (
        "reg_remainder",
        "Define solve(payload). payload has non-negative integer field a and positive integer field b. Return the remainder of a divided by b.",
        [({"a": 7, "b": 3}, 1), ({"a": 9, "b": 3}, 0), ({"a": 5, "b": 5}, 0)],
        'def solve(payload):\n    return payload["a"] % payload["b"]',
    ),
    (
        "reg_power",
        "Define solve(payload). payload has integer field base and non-negative integer field exponent. Return base raised to exponent.",
        [
            ({"base": 2, "exponent": 3}, 8),
            ({"base": 9, "exponent": 0}, 1),
            ({"base": 5, "exponent": 1}, 5),
        ],
        'def solve(payload):\n    return payload["base"] ** payload["exponent"]',
    ),
    (
        "reg_is_even",
        "Define solve(payload). payload is an integer. Return true when it is even, otherwise false.",
        [(4, True), (7, False), (0, True)],
        "def solve(payload):\n    return payload % 2 == 0",
    ),
    (
        "reg_absolute",
        "Define solve(payload). payload is an integer. Return its absolute value.",
        [(-5, 5), (3, 3), (0, 0)],
        "def solve(payload):\n    return abs(payload)",
    ),
    (
        "reg_sign",
        "Define solve(payload). payload is an integer. Return -1 when it is negative, 0 when it is zero, and 1 when it is positive.",
        [(-4, -1), (0, 0), (9, 1)],
        "def solve(payload):\n    if payload < 0:\n        return -1\n    if payload > 0:\n        return 1\n    return 0",
    ),
    (
        "reg_greeting",
        "Define solve(payload). payload is a string holding a name. Return the string Hello, followed by a space, the name, and an exclamation mark.",
        [("Ada", "Hello, Ada!"), ("Bob", "Hello, Bob!"), ("Zo", "Hello, Zo!")],
        'def solve(payload):\n    return "Hello, " + payload + "!"',
    ),
    (
        "reg_key_value_string",
        "Define solve(payload). payload has string fields key and value. Return the string key, an equals sign, then value, with no spaces added.",
        [
            ({"key": "a", "value": "1"}, "a=1"),
            ({"key": "name", "value": "ada"}, "name=ada"),
            ({"key": "", "value": "x"}, "=x"),
        ],
        'def solve(payload):\n    return payload["key"] + "=" + payload["value"]',
    ),
    (
        "reg_pad_zeros",
        "Define solve(payload). payload has string field text and positive integer field width. Return text right-justified to width characters, padded on the left with zeros; longer text is returned unchanged.",
        [
            ({"text": "7", "width": 3}, "007"),
            ({"text": "1234", "width": 2}, "1234"),
            ({"text": "", "width": 2}, "00"),
        ],
        'def solve(payload):\n    return payload["text"].rjust(payload["width"], "0")',
    ),
    (
        "reg_between_bounds",
        "Define solve(payload). payload has integer fields low, high, and value. Return true when value is between low and high inclusive.",
        [
            ({"low": 1, "high": 5, "value": 3}, True),
            ({"low": 1, "high": 5, "value": 5}, True),
            ({"low": 1, "high": 5, "value": 0}, False),
        ],
        'def solve(payload):\n    return payload["low"] <= payload["value"] <= payload["high"]',
    ),
    (
        "reg_sum_of_digits",
        "Define solve(payload). payload is a non-negative integer. Return the sum of its decimal digits.",
        [(123, 6), (0, 0), (999, 27)],
        "def solve(payload):\n    return sum(int(digit) for digit in str(payload))",
    ),
)



def coding_catalog() -> list[CodingTask]:
    regression = [
        _task(
            "reg_sum_even",
            "Define solve(payload). payload is a list of integers. Return the sum of only the even integers.",
            [([1, 2, 3, 4], 6), ([], 0), ([-4, -3, 10], 6)],
            "def solve(payload):\n    return sum(value for value in payload if value % 2 == 0)",
            DatasetRole.REGRESSION,
            family="core_python",
        ),
        _task(
            "reg_dedupe",
            "Define solve(payload). payload is a list of JSON scalar values. Return a list with duplicates removed while preserving first-seen order.",
            [([3, 1, 3, 2, 1], [3, 1, 2]), ([], []), (["a", "a", "b"], ["a", "b"])],
            """def solve(payload):
    result = []
    for value in payload:
        if value not in result:
            result.append(value)
    return result""",
            DatasetRole.REGRESSION,
            family="core_python",
        ),
        _task(
            "reg_run_lengths",
            "Define solve(payload). payload is a string. Return consecutive runs as lists [character, count].",
            [("aaabbc", [["a", 3], ["b", 2], ["c", 1]]), ("", []), ("x", [["x", 1]])],
            """def solve(payload):
    runs = []
    for char in payload:
        if runs and runs[-1][0] == char:
            runs[-1][1] += 1
        else:
            runs.append([char, 1])
    return runs""",
            DatasetRole.REGRESSION,
            family="core_python",
        ),
        _task(
            "reg_balanced",
            "Define solve(payload). payload is a string containing brackets and other characters. Return true when (), [], and {} are correctly nested; ignore other characters.",
            [("a(b[c]{d})", True), ("([)]", False), ("", True), ("]", False)],
            """def solve(payload):
    pairs = {')': '(', ']': '[', '}': '{'}
    stack = []
    for char in payload:
        if char in '([{':
            stack.append(char)
        elif char in pairs:
            if not stack or stack.pop() != pairs[char]:
                return False
    return not stack""",
            DatasetRole.REGRESSION,
            family="core_python",
        ),
    ]
    regression.extend(
        _task(
            task_id,
            prompt,
            cases,
            solution,
            DatasetRole.REGRESSION,
            family="core_python",
        )
        for task_id, prompt, cases, solution in _CORE_REGRESSION_SPECS
    )

    path_get = (
        SPLIT_PATH
        + r"""
def solve(payload):
    node = payload["document"]
    for part in _split(payload["path"]):
        if not isinstance(node, dict) or part not in node:
            return payload.get("default")
        node = node[part]
    return node
"""
    )
    path_exists = (
        SPLIT_PATH
        + r"""
def solve(payload):
    node = payload["document"]
    for part in _split(payload["path"]):
        if not isinstance(node, dict) or part not in node:
            return False
        node = node[part]
    return True
"""
    )
    path_set = (
        SPLIT_PATH
        + r"""
def solve(payload):
    import copy
    result = copy.deepcopy(payload["document"])
    parts = _split(payload["path"])
    node = result
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[parts[-1]] = payload["value"]
    return result
"""
    )
    path_delete = (
        SPLIT_PATH
        + r"""
def solve(payload):
    import copy
    result = copy.deepcopy(payload["document"])
    parts = _split(payload["path"])
    node, parents = result, []
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return result
        parents.append((node, part))
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return result
    del node[parts[-1]]
    for parent, key in reversed(parents):
        if parent[key] == {}:
            del parent[key]
        else:
            break
    return result
"""
    )
    path_flatten = r"""
def _escape(part):
    return part.replace("\\", "\\\\").replace(".", "\\.")

def solve(payload):
    output = {}
    def visit(node, parts):
        if isinstance(node, dict) and node:
            for key in sorted(node):
                visit(node[key], parts + [_escape(key)])
        else:
            output[".".join(parts)] = node
    for key in sorted(payload):
        visit(payload[key], [_escape(key)])
    return output
"""
    path_unflatten = (
        SPLIT_PATH
        + r"""
def solve(payload):
    result = {}
    for path, value in payload.items():
        parts = _split(path)
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result
"""
    )
    training = [
        _task(
            "path_get",
            """Define solve(payload). payload has document, path, and optional default fields. Traverse nested dictionaries using path. Dots separate key segments, but a backslash escapes the next character, so a\\.b means the literal key a.b and x\\\\y means the literal key x\\y. Return default (or null) when missing. Empty key segments are valid.""",
            [
                ({"document": {"a.b": {"c": 3}}, "path": r"a\.b.c"}, 3),
                ({"document": {"a": {"": 4}}, "path": "a."}, 4),
                ({"document": {"x\\y": 8}, "path": r"x\\y"}, 8),
                (
                    {"document": {"a": 1}, "path": "a.b", "default": "missing"},
                    "missing",
                ),
            ],
            path_get,
            DatasetRole.TRAIN,
            family="escaped_path",
        ),
        _task(
            "path_exists",
            """Define solve(payload). Return whether a nested dictionary path exists. Dots separate segments and backslash escapes the next character. Existing values may be null; null still counts as existing. Empty segments are valid.""",
            [
                ({"document": {"a.b": None}, "path": r"a\.b"}, True),
                ({"document": {"a": {"b": 0}}, "path": "a.b"}, True),
                ({"document": {"a": None}, "path": "a.b"}, False),
                ({"document": {}, "path": ""}, False),
            ],
            path_exists,
            DatasetRole.TRAIN,
            family="escaped_path",
        ),
        _task(
            "path_set",
            """Define solve(payload). Return a deep-copied document with value assigned at path, creating dictionary parents. Dots separate segments and backslash escapes the next character. Replace a non-dictionary parent when necessary. Do not mutate payload.""",
            [
                ({"document": {}, "path": r"a\.b.c", "value": 7}, {"a.b": {"c": 7}}),
                ({"document": {"a": 1}, "path": "a.b", "value": 2}, {"a": {"b": 2}}),
                (
                    {"document": {"x": {"y": 1}}, "path": "x.z", "value": 3},
                    {"x": {"y": 1, "z": 3}},
                ),
            ],
            path_set,
            DatasetRole.TRAIN,
            family="escaped_path",
        ),
        _task(
            "path_delete",
            """Define solve(payload). Return a deep-copied document with the given escaped-dot path deleted. Prune parent dictionaries that become empty. Missing paths leave the copy unchanged. Do not mutate payload.""",
            [
                ({"document": {"a.b": {"c": 1}}, "path": r"a\.b.c"}, {}),
                ({"document": {"a": {"b": 1, "c": 2}}, "path": "a.b"}, {"a": {"c": 2}}),
                ({"document": {"a": 1}, "path": "x.y"}, {"a": 1}),
            ],
            path_delete,
            DatasetRole.TRAIN,
            family="escaped_path",
        ),
        _task(
            "path_flatten",
            """Define solve(payload). Flatten a nested dictionary to path:value pairs. Escape backslashes and dots in keys with a backslash. Non-dictionaries and empty dictionaries are leaves. Traverse keys deterministically. payload itself is the document.""",
            [
                ({"a.b": {"c": 1}}, {r"a\.b.c": 1}),
                ({"a": {"x\\y": 2, "z": {}}}, {r"a.x\\y": 2, "a.z": {}}),
                ({"": 4}, {"": 4}),
            ],
            path_flatten,
            DatasetRole.TRAIN,
            family="escaped_path",
        ),
        _task(
            "path_unflatten",
            """Define solve(payload). Expand escaped-dot path:value pairs into a nested dictionary. Dots separate segments and backslash escapes the next character. Empty segments are valid. payload is the flat mapping.""",
            [
                ({r"a\.b.c": 1}, {"a.b": {"c": 1}}),
                ({r"a.x\\y": 2, "a.z": 3}, {"a": {"x\\y": 2, "z": 3}}),
                ({"": 4}, {"": 4}),
            ],
            path_unflatten,
            DatasetRole.TRAIN,
            family="escaped_path",
        ),
    ]

    def variation(source: CodingTask, task_id: str, prompt: str) -> CodingTask:
        return _task(
            task_id,
            prompt,
            [(case.payload, case.expected) for case in source.suite.cases],
            source.reference_solution,
            DatasetRole.TRAIN,
            family="escaped_path",
        )

    # A six-example adapter memorizes operation-shaped prompts too easily.  These
    # independently worded, verifier-backed failures make the learning signal a
    # capability family rather than one canonical prompt per operation.
    variant_prompts = (
        (
            0,
            "path_get_v2",
            r"Write solve(payload) to look up payload['path'] in payload['document']. Split on unescaped dots; backslash quotes the next character. Return payload.get('default') if traversal fails.",
        ),
        (
            0,
            "path_get_v3",
            r"Implement an escaped dotted-key lookup in solve(payload). A dot is a separator unless preceded by backslash. Preserve empty key components and return the optional default for a missing component.",
        ),
        (
            0,
            "path_get_v4",
            r"Create solve(payload) for nested dictionary retrieval. Interpret path with backslash escaping for dots and backslashes; do not treat escaped dots as separators. Missing paths use default or None.",
        ),
        (
            1,
            "path_exists_v2",
            r"Write solve(payload) that reports whether an escaped-dot path is present in document. Backslash escapes the following path character. Presence of a None value must return True.",
        ),
        (
            1,
            "path_exists_v3",
            r"Implement nested-key membership for payload['document'] and payload['path']. Only unescaped dots divide keys, empty keys are legal, and null is an existing value.",
        ),
        (
            2,
            "path_set_v2",
            r"Implement solve(payload) as an immutable escaped-path assignment. Copy document, create dict parents, and place value at path where backslash protects dots and backslashes.",
        ),
        (
            2,
            "path_set_v3",
            r"Write a nested dictionary setter returning a deep copy. Parse only unescaped dots as separators and replace scalar intermediate nodes with dictionaries as needed.",
        ),
        (
            2,
            "path_set_v4",
            r"Define solve(payload) to update the copied document at its escaped dotted path without mutating input. Empty key segments are allowed and missing parents must be created.",
        ),
        (
            3,
            "path_delete_v2",
            r"Write solve(payload) to remove an escaped dotted path from a deep copy of document. Leave a missing path unchanged and prune dictionaries made empty by deletion.",
        ),
        (
            3,
            "path_delete_v3",
            r"Implement immutable nested-key deletion. Backslash escapes the next path character; after removal, recursively discard newly empty ancestor dictionaries.",
        ),
        (
            4,
            "path_flatten_v2",
            r"Define solve(payload) to flatten a nested dictionary. Output dotted paths, escaping every dot or backslash inside an original key. Treat empty dictionaries as leaf values.",
        ),
        (
            4,
            "path_flatten_v3",
            r"Write a deterministic nested-map flattener. Join key components with dots while prefixing literal dots and backslashes in keys with backslash; retain empty dict leaves.",
        ),
        (
            5,
            "path_unflatten_v2",
            r"Implement solve(payload) to rebuild nested dictionaries from flat escaped-dot keys. Unescaped dots split components, backslash quotes the next character, and empty components are valid.",
        ),
        (
            5,
            "path_unflatten_v3",
            r"Expand the input path-to-value mapping into a nested map. Parse backslash-escaped dots and backslashes literally and create all intermediate dictionaries.",
        ),
    )
    training.extend(
        variation(training[source_index], task_id, prompt)
        for source_index, task_id, prompt in variant_prompts
    )

    path_rename = (
        SPLIT_PATH
        + r"""
def _get(document, parts):
    node = document
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node

def solve(payload):
    import copy
    result = copy.deepcopy(payload["document"])
    source, target = _split(payload["from"]), _split(payload["to"])
    exists, value = _get(result, source)
    if not exists:
        return result
    node = result
    for part in source[:-1]:
        node = node[part]
    del node[source[-1]]
    node = result
    for part in target[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[target[-1]] = value
    return result
"""
    )
    path_project = (
        SPLIT_PATH
        + r"""
def solve(payload):
    output = {}
    for path in payload["paths"]:
        node, found = payload["document"], True
        for part in _split(path):
            if not isinstance(node, dict) or part not in node:
                found = False
                break
            node = node[part]
        if found:
            output[path] = node
    return output
"""
    )
    targets = [
        _task(
            "holdout_path_lookup",
            r"Define solve(payload) for dictionary lookup through path. A backslash makes the following character literal, while any other dot separates keys. Return default when a component cannot be followed.",
            [
                ({"document": {"user.name": {"first": "Ada"}}, "path": r"user\.name.first"}, "Ada"),
                ({"document": {"a": {"b\\c": 9}}, "path": r"a.b\\c"}, 9),
                ({"document": {"a": 0}, "path": "a.x", "default": 12}, 12),
            ],
            path_get,
            DatasetRole.TARGET,
            family="escaped_path",
        ),
        _task(
            "holdout_path_membership",
            r"Return from solve(payload) whether document contains the full escaped dotted path. Escaped separators belong to key names and a final null value still counts as present.",
            [
                ({"document": {"x.y": 0}, "path": r"x\.y"}, True),
                ({"document": {"x": {"": None}}, "path": "x."}, True),
                ({"document": {"x": {}}, "path": "x.y"}, False),
            ],
            path_exists,
            DatasetRole.TARGET,
            family="escaped_path",
        ),
        _task(
            "holdout_path_assign",
            r"Deep-copy document and assign value at the backslash-escaped dotted path. Build absent dictionary parents and replace a scalar parent; solve(payload) must not mutate its argument.",
            [
                ({"document": {"a": {}}, "path": r"a.x\.y", "value": 5}, {"a": {"x.y": 5}}),
                ({"document": {"a": 1}, "path": r"a.b\\c", "value": 2}, {"a": {"b\\c": 2}}),
            ],
            path_set,
            DatasetRole.TARGET,
            family="escaped_path",
        ),
        _task(
            "holdout_path_flatten",
            r"Flatten payload's nested dictionaries into dotted paths. Escape literal dots and backslashes in each key, keep empty dictionaries as leaves, and return the flat mapping.",
            [
                ({"a.b": {"c.d": 2}}, {r"a\.b.c\.d": 2}),
                ({"root": {"x\\y": {}}}, {r"root.x\\y": {}}),
            ],
            path_flatten,
            DatasetRole.TARGET,
            family="escaped_path",
        ),
    ]
    future = [
        _task(
            "path_rename",
            """Define solve(payload). Deep-copy document, move the value at escaped-dot path from to escaped-dot path to, creating target parents. If from is missing, return an unchanged copy. Do not mutate payload.""",
            [
                (
                    {"document": {"a.b": 1}, "from": r"a\.b", "to": "x.y"},
                    {"x": {"y": 1}},
                ),
                (
                    {"document": {"a": {"b": 2}}, "from": "a.b", "to": r"c\.d"},
                    {"a": {}, "c.d": 2},
                ),
                ({"document": {"a": 1}, "from": "missing", "to": "x"}, {"a": 1}),
            ],
            path_rename,
            DatasetRole.FUTURE,
            family="escaped_path",
        ),
        _task(
            "path_project",
            """Define solve(payload). payload has document and paths. Return a mapping from each requested escaped-dot path that exists to its value. Preserve the original path strings as output keys; omit missing paths. Null values count as existing.""",
            [
                (
                    {
                        "document": {"a.b": 1, "a": {"c": None}},
                        "paths": [r"a\.b", "a.c", "x"],
                    },
                    {r"a\.b": 1, "a.c": None},
                ),
                ({"document": {}, "paths": []}, {}),
            ],
            path_project,
            DatasetRole.FUTURE,
            family="escaped_path",
        ),
    ]
    return [*regression, *training, *targets, *future]


# ---------------------------------------------------------------------------
# Second growth cycle: the path_restructure failure family.
#
# EXP-005 needs a second failure family that (a) the frozen base demonstrably
# cannot solve, (b) is disjoint by content hash from every cycle-1 training,
# held-out, regression and future prompt, and (c) anchors the transfer targets
# README open question 4 names: path_rename and path_project. Those two tasks
# stay untouched in the FUTURE role -- training on the archived prompts would
# destroy the future-stream probe instead of answering it -- so this family
# trains the *operations* they need (move, copy, pick, drop over escaped-dot
# paths) on freshly authored prompts and cases, and the after-cycle-2 future
# probe then measures transfer to the archived prompts as a diagnostic.
#
# The family deliberately shares the escaped-dot vocabulary with cycle 1's
# escaped_path family. That overlap is the router stressor EXP-005 exists to
# measure, not an accident: a router that confuses the two families records a
# route falsification, not an infrastructure failure.
# ---------------------------------------------------------------------------

_MOVE_SOLUTION = (
    SPLIT_PATH
    + r"""
def solve(payload):
    import copy
    result = copy.deepcopy(payload["document"])
    source, target = _split(payload["from"]), _split(payload["to"])
    node = result
    for part in source:
        if not isinstance(node, dict) or part not in node:
            return result
        node = node[part]
    value = node
    node = result
    for part in source[:-1]:
        node = node[part]
    del node[source[-1]]
    node = result
    for part in target[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[target[-1]] = value
    return result
"""
)

_COPY_SOLUTION = (
    SPLIT_PATH
    + r"""
def solve(payload):
    import copy
    result = copy.deepcopy(payload["document"])
    source, target = _split(payload["from"]), _split(payload["to"])
    node = result
    for part in source:
        if not isinstance(node, dict) or part not in node:
            return result
        node = node[part]
    value = copy.deepcopy(node)
    node = result
    for part in target[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            child = {}
            node[part] = child
        node = child
    node[target[-1]] = value
    return result
"""
)

_PICK_SOLUTION = (
    SPLIT_PATH
    + r"""
def solve(payload):
    output = {}
    for path in payload["paths"]:
        node, found = payload["document"], True
        for part in _split(path):
            if not isinstance(node, dict) or part not in node:
                found = False
                break
            node = node[part]
        if found:
            output[path] = node
    return output
"""
)

_DROP_SOLUTION = (
    SPLIT_PATH
    + r"""
def _delete(result, parts):
    node, parents = result, []
    for part in parts[:-1]:
        if not isinstance(node, dict) or part not in node:
            return
        parents.append((node, part))
        node = node[part]
    if not isinstance(node, dict) or parts[-1] not in node:
        return
    del node[parts[-1]]
    for parent, key in reversed(parents):
        if parent[key] == {}:
            del parent[key]
        else:
            break

def solve(payload):
    import copy
    result = copy.deepcopy(payload["document"])
    for path in payload["paths"]:
        _delete(result, _split(path))
    return result
"""
)

_MOVE_CASES: list[tuple[object, object]] = [
    (
        {"document": {"cfg.db": {"host": "local"}}, "from": r"cfg\.db.host", "to": "server.name"},
        {"cfg.db": {}, "server": {"name": "local"}},
    ),
    ({"document": {"a": {"b": 2}}, "from": "a.b", "to": r"k\.v"}, {"a": {}, "k.v": 2}),
    ({"document": {"x": 5}, "from": "absent.path", "to": "y"}, {"x": 5}),
    ({"document": {"m": {"n\\o": 7}}, "from": r"m.n\\o", "to": "m.p"}, {"m": {"p": 7}}),
]

_COPY_CASES: list[tuple[object, object]] = [
    ({"document": {"a.b": 3}, "from": r"a\.b", "to": "c.d"}, {"a.b": 3, "c": {"d": 3}}),
    (
        {"document": {"src": {"k": None}}, "from": "src.k", "to": "dst.k"},
        {"src": {"k": None}, "dst": {"k": None}},
    ),
    ({"document": {"a": 1}, "from": "missing", "to": "b"}, {"a": 1}),
    (
        {"document": {"t": {"u": {"v": 4}}}, "from": "t.u", "to": r"w\.x"},
        {"t": {"u": {"v": 4}}, "w.x": {"v": 4}},
    ),
]

_PICK_CASES: list[tuple[object, object]] = [
    (
        {
            "document": {"env.prod": {"url": "u1"}, "env": {"dev": "u2"}},
            "paths": [r"env\.prod.url", "env.dev", "env.stage"],
        },
        {r"env\.prod.url": "u1", "env.dev": "u2"},
    ),
    ({"document": {"k": None}, "paths": ["k"]}, {"k": None}),
    ({"document": {}, "paths": ["a", "b"]}, {}),
    ({"document": {"x\\y": 9}, "paths": [r"x\\y"]}, {r"x\\y": 9}),
]

_DROP_CASES: list[tuple[object, object]] = [
    ({"document": {"a.b": {"c": 1}, "d": 2}, "paths": [r"a\.b.c"]}, {"d": 2}),
    ({"document": {"a": {"b": 1, "c": 2}}, "paths": ["a.b", "zz"]}, {"a": {"c": 2}}),
    ({"document": {"m": {"n": {"o": 3}}, "p": 4}, "paths": ["m.n.o", "p"]}, {}),
    ({"document": {"x": 1}, "paths": []}, {"x": 1}),
]

# Independently worded prompts per operation, exactly as cycle 1 did for the
# escaped_path family: one canonical statement plus paraphrases sharing the
# same hidden cases and canonical solution, so the learning signal is the
# operation rather than one prompt string.
_RESTRUCTURE_TRAIN_SPECS: tuple[
    tuple[str, str, list[tuple[object, object]], str], ...
] = (
    (
        "restruct_move",
        "Define solve(payload). payload has document, from, and to. Deep-copy document, remove the value at escaped-dot path from, and insert it at escaped-dot path to, creating dictionary parents along to. Dots separate segments and a backslash escapes the next character. If from is missing, return the unchanged copy. Do not prune emptied parents. Do not mutate payload.",
        _MOVE_CASES,
        _MOVE_SOLUTION,
    ),
    (
        "restruct_move_v2",
        r"Write solve(payload) that relocates one nested value: delete the entry at payload['from'] and store it under payload['to'], both escaped dotted paths where backslash quotes the next character. A missing source leaves the deep copy unchanged; emptied source parents stay in place.",
        _MOVE_CASES,
        _MOVE_SOLUTION,
    ),
    (
        "restruct_move_v3",
        r"Implement an immutable rename over nested dictionaries. Resolve the source path, cut it out, and graft its value at the destination path, building intermediate dictionaries. Only unescaped dots split segments. Return the copy untouched when the source does not exist.",
        _MOVE_CASES,
        _MOVE_SOLUTION,
    ),
    (
        "restruct_move_v4",
        r"Create solve(payload) to transplant a value between two escaped dotted locations of a copied document. Backslash protects dots and backslashes inside key names. Destination parents are created on demand; a vanished source means no change; parents left empty are kept.",
        _MOVE_CASES,
        _MOVE_SOLUTION,
    ),
    (
        "restruct_move_v5",
        r"Define solve(payload) performing a nested-key move: read document, from, and to. Detach the value found at from and reattach it at to without mutating the input. Escaped separators belong to key names. Absent sources return the plain deep copy.",
        _MOVE_CASES,
        _MOVE_SOLUTION,
    ),
    (
        "restruct_copy",
        "Define solve(payload). payload has document, from, and to. Deep-copy document and duplicate the value at escaped-dot path from into escaped-dot path to, creating dictionary parents along to and leaving the source in place. Dots separate segments and a backslash escapes the next character. If from is missing, return the unchanged copy. Do not mutate payload.",
        _COPY_CASES,
        _COPY_SOLUTION,
    ),
    (
        "restruct_copy_v2",
        r"Write solve(payload) that mirrors one nested value: keep the entry at payload['from'] and also store an independent copy of it under payload['to']. Both are escaped dotted paths; backslash quotes the next character. A missing source leaves the deep copy unchanged.",
        _COPY_CASES,
        _COPY_SOLUTION,
    ),
    (
        "restruct_copy_v3",
        r"Implement an immutable duplication over nested dictionaries: resolve the source path and graft a deep copy of its value at the destination path, building intermediate dictionaries as needed. Only unescaped dots split segments. The original entry survives.",
        _COPY_CASES,
        _COPY_SOLUTION,
    ),
    (
        "restruct_copy_v4",
        r"Create solve(payload) to replicate a value between two escaped dotted locations of a copied document, source retained. Backslash protects dots and backslashes inside key names. Destination parents are created on demand; a vanished source means no change.",
        _COPY_CASES,
        _COPY_SOLUTION,
    ),
    (
        "restruct_copy_v5",
        r"Define solve(payload) performing a nested-key copy from document at from to destination to, without mutating the input and without detaching the source. Escaped separators belong to key names. Absent sources return the plain deep copy.",
        _COPY_CASES,
        _COPY_SOLUTION,
    ),
    (
        "restruct_pick",
        "Define solve(payload). payload has document and paths. Return a mapping from each escaped-dot path in paths that exists in document to its value, keyed by the original path string. Omit missing paths; null values count as existing. Dots separate segments and a backslash escapes the next character.",
        _PICK_CASES,
        _PICK_SOLUTION,
    ),
    (
        "restruct_pick_v2",
        r"Write solve(payload) that selects a subset of a nested document: for every requested escaped dotted path that resolves, emit path -> resolved value in the output dictionary. Requests that cannot be followed are simply skipped, and a stored None still resolves.",
        _PICK_CASES,
        _PICK_SOLUTION,
    ),
    (
        "restruct_pick_v3",
        r"Implement a projection over nested dictionaries. Walk each path in payload['paths'] through payload['document'], splitting only on unescaped dots, and collect the survivors into a flat result keyed by the untouched request strings.",
        _PICK_CASES,
        _PICK_SOLUTION,
    ),
    (
        "restruct_pick_v4",
        r"Create solve(payload) returning only the requested entries of a document: resolve every escaped dotted request, keep the ones that exist (null included), and drop the rest. Backslash protects dots and backslashes inside key names.",
        _PICK_CASES,
        _PICK_SOLUTION,
    ),
    (
        "restruct_pick_v5",
        r"Define solve(payload) to extract chosen values from nested dictionaries: for each path string given, follow its escaped-dot segments and, when the walk succeeds, record the original string mapped to the value found.",
        _PICK_CASES,
        _PICK_SOLUTION,
    ),
    (
        "restruct_drop",
        "Define solve(payload). payload has document and paths. Deep-copy document and delete the entry at every escaped-dot path in paths, pruning parent dictionaries that become empty. Missing paths are ignored. Dots separate segments and a backslash escapes the next character. Do not mutate payload.",
        _DROP_CASES,
        _DROP_SOLUTION,
    ),
    (
        "restruct_drop_v2",
        r"Write solve(payload) that strips several nested entries at once: remove each escaped dotted path in payload['paths'] from a deep copy of payload['document'], discarding ancestor dictionaries emptied by a deletion. Paths that do not resolve change nothing.",
        _DROP_CASES,
        _DROP_SOLUTION,
    ),
    (
        "restruct_drop_v3",
        r"Implement an immutable bulk deletion over nested dictionaries. For every listed path, splitting only on unescaped dots, cut the addressed entry out of the copy and recursively drop newly empty parents. Unknown paths are skipped.",
        _DROP_CASES,
        _DROP_SOLUTION,
    ),
    (
        "restruct_drop_v4",
        r"Create solve(payload) to erase a list of escaped dotted locations from a copied document. Backslash protects dots and backslashes inside key names. After each removal, ancestors left with no entries are deleted too; absent locations are ignored.",
        _DROP_CASES,
        _DROP_SOLUTION,
    ),
    (
        "restruct_drop_v5",
        r"Define solve(payload) performing multi-path removal: apply every deletion request in order to one deep copy, pruning dictionaries a removal empties, and return the result without mutating the input.",
        _DROP_CASES,
        _DROP_SOLUTION,
    ),
)

# Fresh held-out prompts authored for EXP-005 and never read by any prior
# gate, training run, or tuning decision. One per operation, with their own
# hidden cases.
_RESTRUCTURE_TARGET_SPECS: tuple[
    tuple[str, str, list[tuple[object, object]], str], ...
] = (
    (
        "holdout_restruct_move",
        r"Deep-copy document and move the value stored at the backslash-escaped dotted path from so it lives at path to instead, creating destination parents. Leave the copy unchanged when from is absent; solve(payload) must not mutate its argument.",
        [
            (
                {"document": {"log.dir": "/tmp"}, "from": r"log\.dir", "to": "paths.log"},
                {"paths": {"log": "/tmp"}},
            ),
            ({"document": {"a": {"b": 5}}, "from": "a.b", "to": "a.c"}, {"a": {"c": 5}}),
            ({"document": {"q": 1}, "from": "nope", "to": "r"}, {"q": 1}),
        ],
        _MOVE_SOLUTION,
    ),
    (
        "holdout_restruct_copy",
        r"Duplicate the entry found at one escaped dotted path of document into another path of a deep copy, keeping the original entry. Build missing destination parents; an unresolvable source returns the untouched copy from solve(payload).",
        [
            (
                {"document": {"tpl.base": {"w": 1}}, "from": r"tpl\.base", "to": "out"},
                {"tpl.base": {"w": 1}, "out": {"w": 1}},
            ),
            ({"document": {"a": 2}, "from": "a", "to": r"b\.c"}, {"a": 2, "b.c": 2}),
        ],
        _COPY_SOLUTION,
    ),
    (
        "holdout_restruct_pick",
        r"From solve(payload), return the requested slice of document: each escaped dotted path in paths that resolves maps, under its original string, to the value it reaches. Unresolvable requests are omitted and null values are kept.",
        [
            (
                {
                    "document": {"u.id": 7, "u": {"name": "kim"}},
                    "paths": [r"u\.id", "u.name", "u.mail"],
                },
                {r"u\.id": 7, "u.name": "kim"},
            ),
            ({"document": {"z": None}, "paths": ["z", "y"]}, {"z": None}),
        ],
        _PICK_SOLUTION,
    ),
    (
        "holdout_restruct_drop",
        r"Remove every listed backslash-escaped dotted path from a deep copy of document, pruning dictionaries a deletion leaves empty, and return the copy from solve(payload). Requests that do not resolve are ignored.",
        [
            (
                {"document": {"tmp.cache": {"k": 1}, "keep": True}, "paths": [r"tmp\.cache.k"]},
                {"keep": True},
            ),
            ({"document": {"a": {"b": 1}}, "paths": ["a.c"]}, {"a": {"b": 1}}),
        ],
        _DROP_SOLUTION,
    ),
)


def second_cycle_catalog() -> list[CodingTask]:
    """The path_restructure family for EXP-005's second growth cycle.

    Kept out of ``coding_catalog()`` so every single-cycle run keeps exactly
    the cohort identity it always had; only a multi-cycle run merges this in.
    """
    training = [
        _task(
            task_id,
            prompt,
            cases,
            solution,
            DatasetRole.TRAIN,
            family="path_restructure",
        )
        for task_id, prompt, cases, solution in _RESTRUCTURE_TRAIN_SPECS
    ]
    targets = [
        _task(
            task_id,
            prompt,
            cases,
            solution,
            DatasetRole.TARGET,
            family="path_restructure",
        )
        for task_id, prompt, cases, solution in _RESTRUCTURE_TARGET_SPECS
    ]
    return [*training, *targets]
