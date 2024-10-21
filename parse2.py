#!/usr/bin/env python3
"""
Determine the highest-probability (minimum-weight) parse of sentences under a CFG,
using a probabilistic Earley's algorithm.
"""

from __future__ import annotations
import argparse
import logging
import math
import tqdm
import heapq
from dataclasses import dataclass, field
from pathlib import Path
from collections import Counter, defaultdict
from typing import Counter as CounterType, Iterable, List, Optional, Dict, Tuple, Any

log = logging.getLogger(Path(__file__).stem)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "grammar", type=Path, help="Path to .gr file containing a PCFG"
    )
    parser.add_argument(
        "sentences", type=Path, help="Path to .sen file containing tokenized input sentences"
    )
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )

    parser.add_argument(
        "--progress",
        action="store_true",
        help="Display a progress bar",
        default=False,
    )

    # for verbosity of logging
    parser.set_defaults(logging_level=logging.INFO)
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", dest="logging_level", action="store_const", const=logging.DEBUG
    )
    verbosity.add_argument(
        "-q", "--quiet",   dest="logging_level", action="store_const", const=logging.WARNING
    )

    return parser.parse_args()


class EarleyChart:
    """A chart for Earley's algorithm."""

    def __init__(self, tokens: List[str], grammar: Grammar, progress: bool = False) -> None:
        """Create the chart based on parsing `tokens` with `grammar`.
        `progress` says whether to display progress bars as we parse."""
        self.tokens = tokens
        self.grammar = grammar
        self.progress = progress
        self.profile: CounterType[str] = Counter()

        self.cols: List[Agenda]
        self.found_goal_item: Optional[Item] = None  # To store the final goal item
        self._run_earley()    # run Earley's algorithm to construct self.cols

    def accepted(self) -> bool:
        """Was the sentence accepted?
        That is, does the finished chart contain an item corresponding to a parse of the sentence?
        This method answers the recognition question, but not the parsing question."""
        return self.found_goal_item is not None

    def get_best_parse(self) -> Optional[Tuple[Any, float]]:
        """Return the best parse tree and its weight, or None if no parse exists."""
        if not self.accepted():
            return None
        else:
            return (self._build_tree(self.found_goal_item), self.found_goal_item.weight)

    def _build_tree(self, item: Item) -> Any:
        """Recursively build the parse tree from the backpointers."""
        if not item.backpointers:
            # Should not happen, but in case
            return (item.rule.lhs,)
        else:
            children = []
            for bp in item.backpointers:
                if isinstance(bp, Item):
                    child = self._build_tree(bp)
                else:
                    # Terminal token
                    child = bp
                children.append(child)
            return (item.rule.lhs, *children)

    def _run_earley(self) -> None:
        """Fill in the Earley chart."""
        # Initially empty column for each position in sentence
        self.cols = [Agenda() for _ in range(len(self.tokens) + 1)]

        # Start looking for ROOT at position 0
        self._predict(self.grammar.start_symbol, 0)

        # Process the columns
        for i, column in tqdm.tqdm(enumerate(self.cols),
                                   total=len(self.cols),
                                   disable=not self.progress):
            log.debug("")
            log.debug(f"Processing items in column {i}")
            while column:
                item = column.pop()
                next_sym = item.next_symbol()
                if next_sym is None:
                    # Attach this complete constituent to its customers
                    log.debug(f"{item} => ATTACH")
                    self._attach(item, i)
                    # Check if this is a completed ROOT spanning the whole input
                    if (item.rule.lhs == self.grammar.start_symbol and
                        item.start_position == 0 and
                        i == len(self.tokens)):
                        # Found a complete parse
                        if (self.found_goal_item is None or
                            item.weight < self.found_goal_item.weight):
                            self.found_goal_item = item
                elif self.grammar.is_nonterminal(next_sym):
                    # Predict the nonterminal after the dot
                    log.debug(f"{item} => PREDICT")
                    self._predict(next_sym, i)
                else:
                    # Try to scan the terminal after the dot
                    log.debug(f"{item} => SCAN")
                    self._scan(item, i)

    def _predict(self, nonterminal: str, position: int) -> None:
        """Start looking for this nonterminal at the given position."""
        for rule in self.grammar.expansions(nonterminal):
            new_item = Item(rule=rule, dot_position=0, start_position=position, weight=rule.weight)
            self.cols[position].push(new_item)
            log.debug(f"\tPredicted: {new_item} in column {position}")
            self.profile["PREDICT"] += 1

    def _scan(self, item: Item, position: int) -> None:
        """Attach the next word to this item that ends at position,
        if it matches what this item is looking for next."""
        if position < len(self.tokens) and self.tokens[position] == item.next_symbol():
            new_backpointers = item.backpointers + [self.tokens[position]]
            new_item = item.advance(dot_position=item.dot_position + 1,
                                    weight=item.weight,
                                    backpointers=new_backpointers)
            self.cols[position + 1].push(new_item)
            log.debug(f"\tScanned to get: {new_item} in column {position+1}")
            self.profile["SCAN"] += 1

    def _attach(self, item: Item, position: int) -> None:
        """Attach this complete item to its customers in previous columns, advancing the
        customers' dots to create new items in this column."""
        mid = item.start_position
        for customer in self.cols[mid].all():
            if customer.next_symbol() == item.rule.lhs:
                new_weight = customer.weight + item.weight
                new_backpointers = customer.backpointers + [item]
                new_item = customer.advance(dot_position=customer.dot_position + 1,
                                            weight=new_weight,
                                            backpointers=new_backpointers)
                self.cols[position].push(new_item)
                log.debug(f"\tAttached to get: {new_item} in column {position}")
                self.profile["ATTACH"] += 1


class Agenda:
    """An agenda of items that need to be processed."""

    def __init__(self) -> None:
        self._heap: List[Tuple[float, int, Item]] = []
        self._index: Dict[ItemKey, Item] = {}
        self._counter = 0  # Unique sequence count to avoid comparison issues

    def __len__(self) -> int:
        """Returns number of items that are still waiting to be popped."""
        return len(self._heap)

    def push(self, item: Item) -> None:
        """Add (enqueue) the item, handling duplicates with lower weights."""
        key = item.get_key()
        existing_item = self._index.get(key)
        if existing_item is None:
            # New item
            heapq.heappush(self._heap, (item.weight, self._counter, item))
            self._index[key] = item
            self._counter += 1
        elif item.weight < existing_item.weight:
            # Found a better (lower weight) item
            self._index[key] = item
            # Re-insert the item into the heap for reprocessing
            heapq.heappush(self._heap, (item.weight, self._counter, item))
            self._counter += 1

    def pop(self) -> Item:
        """Returns one of the items that was waiting to be popped (dequeued)."""
        if len(self) == 0:
            raise IndexError
        _, _, item = heapq.heappop(self._heap)
        return item

    def all(self) -> Iterable[Item]:
        """Collection of all items that have ever been pushed."""
        return self._index.values()

    def __repr__(self):
        """Provide a human-readable string representation of this Agenda."""
        return f"{self.__class__.__name__}({self._heap})"


class Grammar:
    """Represents a weighted context-free grammar."""

    def __init__(self, start_symbol: str, *files: Path) -> None:
        """Create a grammar with the given start symbol,
        adding rules from the specified files if any."""
        self.start_symbol = start_symbol
        self._expansions: Dict[str, List[Rule]] = defaultdict(list)
        # Read the input grammar files
        for file in files:
            self.add_rules_from_file(file)

    def add_rules_from_file(self, file: Path) -> None:
        """Add rules to this grammar from a file (one rule per line).
        Each rule is preceded by a normalized probability p,
        and we take -log2(p) to be the rule's weight."""
        with open(file, "r") as f:
            for line in f:
                # remove any comment from end of line, and any trailing whitespace
                line = line.split("#")[0].rstrip()
                # skip empty lines
                if line == "":
                    continue
                # Parse tab-delimited line of format <probability>\t<lhs>\t<rhs>
                _prob, lhs, _rhs = line.split("\t")
                prob = float(_prob)
                rhs = tuple(_rhs.split())
                weight = -math.log2(prob)
                rule = Rule(lhs=lhs, rhs=rhs, weight=weight)
                self._expansions[lhs].append(rule)

    def expansions(self, lhs: str) -> Iterable[Rule]:
        """Return an iterable collection of all rules with a given lhs"""
        return self._expansions[lhs]

    def is_nonterminal(self, symbol: str) -> bool:
        """Is symbol a nonterminal symbol?"""
        return symbol in self._expansions


@dataclass(frozen=True)
class Rule:
    """
    A grammar rule has a left-hand side (lhs), a right-hand side (rhs), and a weight.
    """
    lhs: str
    rhs: Tuple[str, ...]
    weight: float = 0.0

    def __repr__(self) -> str:
        """Complete string used to show this rule instance at the command line"""
        return f"{self.lhs} → {' '.join(self.rhs)}"


@dataclass(frozen=True)
class ItemKey:
    """A key for items, used to detect duplicates."""
    rule: Rule
    dot_position: int
    start_position: int

@dataclass
class Item:
    """An item in the Earley parse chart, representing one or more subtrees
    that could yield a particular substring."""
    rule: Rule
    dot_position: int
    start_position: int
    weight: float
    backpointers: List[Any] = field(default_factory=list)  # backpointers can include Items or tokens

    def next_symbol(self) -> Optional[str]:
        """What's the next, unprocessed symbol (terminal, non-terminal, or None) in this partially matched rule?"""
        assert 0 <= self.dot_position <= len(self.rule.rhs)
        if self.dot_position == len(self.rule.rhs):
            return None
        else:
            return self.rule.rhs[self.dot_position]

    def advance(self, dot_position: int, weight: float, backpointers: List[Any]) -> Item:
        return Item(rule=self.rule,
                    dot_position=dot_position,
                    start_position=self.start_position,
                    weight=weight,
                    backpointers=backpointers)

    def get_key(self) -> ItemKey:
        """Return a hashable key for this item, used for duplicate detection."""
        return ItemKey(self.rule, self.dot_position, self.start_position)

    def __hash__(self):
        return hash((self.rule, self.dot_position, self.start_position))

    def __eq__(self, other):
        return (self.rule == other.rule and
                self.dot_position == other.dot_position and
                self.start_position == other.start_position)

    def __repr__(self) -> str:
        """Human-readable representation string used when printing this item."""
        DOT = "·"
        rhs = list(self.rule.rhs)  # Make a copy.
        rhs.insert(self.dot_position, DOT)
        dotted_rule = f"{self.rule.lhs} → {' '.join(rhs)}"
        return f"({self.start_position}, {dotted_rule}, w={self.weight:.3f})"


def format_tree(tree) -> str:
    """Format the parse tree in the required output format."""
    if isinstance(tree, tuple):
        label = tree[0]
        children = tree[1:]
        if len(children) == 1 and isinstance(children[0], str):
            # Terminal node
            return f"({label} {children[0]})"
        else:
            # Non-terminal node
            children_str = ' '.join(format_tree(child) for child in children)
            return f"({label} {children_str})"
    else:
        # Should not happen
        return str(tree)


def main():
    # Parse the command-line arguments
    args = parse_args()
    logging.basicConfig(level=args.logging_level)

    grammar = Grammar(args.start_symbol, args.grammar)

    with open(args.sentences) as f:
        for sentence in f:
            sentence = sentence.strip()
            if sentence != "":  # skip blank lines
                # Analyze the sentence
                log.debug("="*70)
                log.debug(f"Parsing sentence: {sentence}")
                tokens = sentence.split()
                chart = EarleyChart(tokens, grammar, progress=args.progress)
                # Print the result
                result = chart.get_best_parse()
                if result is None:
                    print("NONE")
                else:
                    tree, weight = result
                    print(format_tree(tree))
                    print(f"{weight}")
                log.debug(f"Profile of work done: {chart.profile}")


if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=False)   # run tests
    main()
