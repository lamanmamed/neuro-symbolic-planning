"""End-to-end object grounding and symbolic plan generation."""

from pathlib import Path
from typing import Optional

import torch

from .perception import predict_object
from .planning import CIFAR_100_CLASSES, create_problem, solve_problem


def normalize_object_name(name: str) -> str:
    """Convert CIFAR labels and user text to the PDDL naming convention."""
    return name.strip().lower().replace("_", "-").replace(" ", "-")


def _ground(predicates: list[str], object_name: str) -> list[str]:
    return [predicate.replace("?x", object_name) for predicate in predicates]


def generate_plan(
    input_data: torch.Tensor | str,
    initial_state: list[str],
    goal_state: list[str],
    *,
    domain_file: str | Path,
    base_problem: str | Path,
    projection_checkpoint: str | Path,
) -> Optional[list[str]]:
    """Identify an object, ground it into PDDL predicates, and search for a plan."""
    if isinstance(input_data, torch.Tensor):
        predictions = predict_object(input_data, projection_checkpoint, top_k=1)
        if not predictions:
            return None
        object_name = normalize_object_name(predictions[0][0])
    elif isinstance(input_data, str):
        object_name = normalize_object_name(input_data)
    else:
        raise TypeError("input_data must be a text label or image tensor.")

    if object_name not in CIFAR_100_CLASSES:
        raise ValueError(f"Unknown CIFAR-100 object: {object_name}")

    grounded_initial = _ground(initial_state, object_name)
    grounded_goals = _ground(goal_state, object_name)

    # Supply the small set of assumptions the manipulation actions need.
    object_location = None
    for predicate in grounded_initial:
        parts = predicate.strip().strip("()").split()
        if len(parts) == 3 and parts[0] == "at" and parts[1] == object_name:
            object_location = parts[2]
            break

    if object_location is None:
        object_location = "lab"
        grounded_initial.append(f"(at {object_name} {object_location})")

    grounded_initial.extend([
        f"(agent-at {object_location})",
        "(hand-empty)",
        f"(clear {object_name})",
    ])

    problem_file = create_problem(
        base_problem,
        initial_overrides=set(grounded_initial),
        goals=set(grounded_goals),
        name="neuro-symbolic-demo",
    )
    return solve_problem(domain_file, problem_file)
