"""Small PDDL parser, action grounder and heuristic A* planner."""

from __future__ import annotations

import heapq
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import FrozenSet, Optional


CIFAR_100_CLASSES = {
    "apple", "aquarium-fish", "baby", "bear", "beaver", "bed", "bee", "beetle",
    "bicycle", "bottle", "bowl", "boy", "bridge", "bus", "butterfly", "camel",
    "can", "castle", "caterpillar", "cattle", "chair", "chimpanzee", "clock",
    "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster",
    "house", "kangaroo", "keyboard", "lamp", "lawn-mower", "leopard", "lion",
    "lizard", "lobster", "man", "maple", "motorcycle", "mountain", "mouse",
    "mushroom", "oak", "orange", "orchid", "otter", "palm", "pear",
    "pickup-truck", "pine", "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket", "rose", "sea", "seal",
    "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet-pepper", "table", "tank",
    "telephone", "television", "tiger", "tractor", "train", "trout", "tulip",
    "turtle", "wardrobe", "whale", "willow", "wolf", "woman", "worm",
}
TOOLS = {"knife", "dslr"}
LOCATIONS = {"lab", "outdoors"}


@dataclass(frozen=True)
class Predicate:
    name: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        return f"({self.name} {' '.join(self.args)})" if self.args else f"({self.name})"

    @staticmethod
    def from_string(value: str) -> "Predicate":
        parts = value.strip().strip("()").split()
        if not parts:
            raise ValueError("Empty predicate.")
        return Predicate(parts[0], tuple(parts[1:]))


@dataclass(frozen=True)
class Action:
    name: str
    parameters: tuple[str, ...]
    preconditions: FrozenSet[Predicate]
    add_effects: FrozenSet[Predicate]
    del_effects: FrozenSet[Predicate]

    def __str__(self) -> str:
        return f"({self.name} {' '.join(self.parameters)})"

    def instantiate(self, bindings: dict[str, str]) -> "Action":
        def bind(predicate: Predicate) -> Predicate:
            return Predicate(predicate.name, tuple(bindings.get(arg, arg) for arg in predicate.args))

        return Action(
            self.name,
            tuple(bindings.get(parameter, parameter) for parameter in self.parameters),
            frozenset(bind(p) for p in self.preconditions),
            frozenset(bind(p) for p in self.add_effects),
            frozenset(bind(p) for p in self.del_effects),
        )


@dataclass(frozen=True)
class State:
    predicates: FrozenSet[Predicate]

    def is_applicable(self, action: Action) -> bool:
        return action.preconditions.issubset(self.predicates)

    def apply(self, action: Action) -> "State":
        predicates = set(self.predicates)
        predicates.difference_update(action.del_effects)
        predicates.update(action.add_effects)
        return State(frozenset(predicates))

    def satisfies(self, goal: FrozenSet[Predicate]) -> bool:
        return goal.issubset(self.predicates)


@dataclass(order=True)
class SearchNode:
    f_score: int
    state: State = field(compare=False)
    action: Optional[Action] = field(compare=False)
    parent: Optional["SearchNode"] = field(compare=False)
    g_score: int = field(compare=False)

    def plan(self) -> list[Action]:
        actions = []
        node = self
        while node.parent is not None:
            actions.append(node.action)
            node = node.parent
        return list(reversed(actions))


class PDDLParser:
    """Parser for the small STRIPS subset used by this planning domain."""

    @staticmethod
    def _extract_predicates(block: str) -> set[Predicate]:
        predicates: set[Predicate] = set()
        depth = 0
        current: list[str] = []

        for char in block:
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
                if depth == 0 and current:
                    text = "".join(current).strip()
                    if text.startswith("(and"):
                        inner = text[4:-1]
                        predicates.update(PDDLParser._extract_predicates(inner))
                    elif not text.startswith("(not") and not text.startswith("(forall"):
                        predicates.add(Predicate.from_string(text))
                    current = []
            elif depth > 0:
                current.append(char)

        return predicates

    @staticmethod
    def _parse_effects(body: str) -> tuple[set[Predicate], set[Predicate]]:
        match = re.search(r":effect\s+(\(.*)", body, re.DOTALL | re.IGNORECASE)
        if not match:
            return set(), set()

        block = match.group(1).strip()
        depth = 0
        end = len(block)
        for index, char in enumerate(block):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        block = block[:end]
        if block.startswith("(and"):
            block = block[4:-1]

        additions: set[Predicate] = set()
        deletions: set[Predicate] = set()
        depth = 0
        current: list[str] = []

        for char in block:
            if char == "(":
                depth += 1
                current.append(char)
            elif char == ")":
                depth -= 1
                current.append(char)
                if depth == 0 and current:
                    text = "".join(current).strip()
                    if text.startswith("(not"):
                        inner = re.search(r"\(not\s+(\(.*?\))\s*\)", text, re.IGNORECASE)
                        if inner:
                            deletions.add(Predicate.from_string(inner.group(1)))
                    else:
                        additions.add(Predicate.from_string(text))
                    current = []
            elif depth > 0:
                current.append(char)

        return additions, deletions

    @staticmethod
    def parse_domain(path: str | Path) -> dict[str, Action]:
        content = Path(path).read_text()
        actions: dict[str, Action] = {}

        for match in re.finditer(r"\(:action\s+(\S+)(.*?)(?=\(:action|\Z)", content, re.DOTALL | re.IGNORECASE):
            name, body = match.groups()
            parameters: list[str] = []

            parameter_match = re.search(r":parameters\s+\((.*?)\)", body, re.DOTALL | re.IGNORECASE)
            if parameter_match:
                parameters = re.findall(r"\?[\w-]+", parameter_match.group(1))

            preconditions: set[Predicate] = set()
            precondition_match = re.search(r":precondition\s+(\(.*?)(?=\s*:effect|\Z)", body, re.DOTALL | re.IGNORECASE)
            if precondition_match:
                preconditions = PDDLParser._extract_predicates(precondition_match.group(1))

            additions, deletions = PDDLParser._parse_effects(body)
            actions[name] = Action(
                name,
                tuple(parameters),
                frozenset(preconditions),
                frozenset(additions),
                frozenset(deletions),
            )

        return actions

    @staticmethod
    def parse_problem(path: str | Path) -> tuple[dict[str, set[str]], State, FrozenSet[Predicate]]:
        content = Path(path).read_text()
        objects: dict[str, set[str]] = defaultdict(set)

        object_match = re.search(r":objects(.*?)(?=\(:init)", content, re.DOTALL | re.IGNORECASE)
        if object_match:
            block = re.sub(r";.*$", "", object_match.group(1), flags=re.MULTILINE)
            for token in re.findall(r"[^\s():;]+", block):
                token = token.lower()
                if token in CIFAR_100_CLASSES:
                    objects["item"].add(token)

        objects["item"].update(TOOLS)
        objects["tool"].update(TOOLS)
        objects["location"].update(LOCATIONS)

        init: set[Predicate] = set()
        init_match = re.search(r":init(.*?)(?=\(:goal|\Z)", content, re.DOTALL | re.IGNORECASE)
        if init_match:
            init = PDDLParser._extract_predicates(init_match.group(1))

        goal: set[Predicate] = set()
        goal_match = re.search(r":goal\s+(\(.*\))", content, re.DOTALL | re.IGNORECASE)
        if goal_match:
            goal = PDDLParser._extract_predicates(goal_match.group(1))

        return dict(objects), State(frozenset(init)), frozenset(goal)


class ActionGrounder:
    def __init__(self, schemas: dict[str, Action], objects: dict[str, set[str]]) -> None:
        self.schemas = schemas
        self.items = sorted(objects.get("item", set()))
        self.cifar_objects = sorted(set(self.items) - TOOLS)
        self.locations = sorted(objects.get("location", LOCATIONS))

    def ground_all(self) -> list[Action]:
        actions: list[Action] = []
        for name, schema in self.schemas.items():
            if name == "walk-between-rooms":
                actions.extend(
                    schema.instantiate({schema.parameters[0]: left, schema.parameters[1]: right})
                    for left in self.locations
                    for right in self.locations
                    if left != right
                )
            elif name in {"pick-up", "put-down"}:
                actions.extend(
                    schema.instantiate({schema.parameters[0]: item, schema.parameters[1]: location})
                    for item in self.items
                    for location in self.locations
                )
            elif name in {"stack", "unstack"}:
                actions.extend(
                    schema.instantiate({
                        schema.parameters[0]: top,
                        schema.parameters[1]: bottom,
                        schema.parameters[2]: location,
                    })
                    for top in self.items
                    for bottom in self.items
                    if top != bottom
                    for location in self.locations
                )
            elif name in {"slice-object", "take-photo"}:
                actions.extend(
                    schema.instantiate({schema.parameters[0]: obj, schema.parameters[1]: location})
                    for obj in self.cifar_objects
                    for location in self.locations
                )
        return actions


def heuristic(state: State, goal: FrozenSet[Predicate]) -> int:
    """Prefer states that satisfy more goals and have the needed tool in hand."""
    score = 0
    for target in goal:
        if target in state.predicates:
            continue
        score += 5
        if target.name == "cut-into-pieces":
            if Predicate("holding", ("knife",)) not in state.predicates:
                score += 2
        elif target.name == "documented":
            if Predicate("holding", ("dslr",)) not in state.predicates:
                score += 2
    return score


def _relevant_actions(actions: list[Action], goal: FrozenSet[Predicate]) -> list[Action]:
    goal_actions = [action for action in actions if set(action.add_effects) & set(goal)]
    supporting = set().union(*(set(action.preconditions) for action in goal_actions)) if goal_actions else set()

    useful = [
        action for action in actions
        if action in goal_actions
        or set(action.add_effects) & supporting
        or action.name in {"walk-between-rooms", "pick-up", "put-down", "unstack"}
    ]
    return useful or actions


def astar_search(
    initial: State,
    goal: FrozenSet[Predicate],
    actions: list[Action],
    max_iterations: int = 80_000,
) -> Optional[list[str]]:
    """Return a plan found by heuristic A* search, or None if the search is exhausted."""
    if initial.satisfies(goal):
        return []

    start = SearchNode(heuristic(initial, goal), initial, None, None, 0)
    frontier = [start]
    best_cost = {initial: 0}
    candidate_actions = _relevant_actions(actions, goal)

    for _ in range(max_iterations):
        if not frontier:
            break

        node = heapq.heappop(frontier)
        if node.state.satisfies(goal):
            return [str(action) for action in node.plan()]

        for action in candidate_actions:
            if not node.state.is_applicable(action):
                continue

            next_state = node.state.apply(action)
            cost = node.g_score + 1
            if cost >= best_cost.get(next_state, float("inf")):
                continue

            best_cost[next_state] = cost
            heapq.heappush(
                frontier,
                SearchNode(cost + heuristic(next_state, goal), next_state, action, node, cost),
            )

    return None


def solve_problem(domain_file: str | Path, problem_file: str | Path) -> Optional[list[str]]:
    schemas = PDDLParser.parse_domain(domain_file)
    objects, initial_state, goal = PDDLParser.parse_problem(problem_file)
    actions = ActionGrounder(schemas, objects).ground_all()
    return astar_search(initial_state, goal, actions)


def _conflicts(base: Predicate, user_predicates: set[Predicate]) -> bool:
    """Return True when a user predicate replaces the same part of the base state."""
    for user in user_predicates:
        if base == user:
            return True
        if base.name == "agent-at" and user.name == "agent-at":
            return True
        if base.args and user.args and base.args[0] == user.args[0]:
            if base.name in {"at", "on-top"} and user.name in {"at", "on-top"}:
                return True
            if base.name == "whole" and user.name == "cut-into-pieces":
                return True
        if base.name == "hand-empty" and user.name == "holding":
            return True
        if base.name == "holding" and user.name == "hand-empty":
            return True
    return False


def create_problem(
    base_problem: str | Path,
    initial_overrides: set[str],
    goals: set[str],
    name: str = "generated",
) -> str:
    """Create a temporary PDDL problem by replacing selected base-state predicates."""
    content = Path(base_problem).read_text()
    objects_match = re.search(r"(\(:objects.*?\)(?=\s*\(:init))", content, re.DOTALL | re.IGNORECASE)
    if not objects_match:
        raise ValueError("Could not find the objects section in the base problem.")

    init_match = re.search(r":init(.*?)(?=\(:goal|\Z)", content, re.DOTALL | re.IGNORECASE)
    base_init = PDDLParser._extract_predicates(init_match.group(1)) if init_match else set()

    user_init = {Predicate.from_string(value) for value in initial_overrides}
    user_goals = {Predicate.from_string(value) for value in goals}

    final_init = {predicate for predicate in base_init if not _conflicts(predicate, user_init)}
    final_init.update(user_init)

    problem = f"""(define (problem {name})
  (:domain cifar100-process)
  {objects_match.group(1)}
  (:init
{chr(10).join(f'    {predicate}' for predicate in sorted(final_init, key=str))}
  )
  (:goal (and
{chr(10).join(f'    {predicate}' for predicate in sorted(user_goals, key=str))}
  ))
)
"""

    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".pddl", delete=False)
    handle.write(problem)
    handle.close()
    return handle.name
