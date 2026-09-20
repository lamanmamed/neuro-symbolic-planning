from pathlib import Path

from neuro_symbolic import generate_plan


ROOT = Path(__file__).resolve().parents[1]

plan = generate_plan(
    "apple",
    initial_state=["(at ?x lab)", "(whole ?x)"],
    goal_state=["(cut-into-pieces ?x)"],
    domain_file=ROOT / "pddl" / "domain.pddl",
    base_problem=ROOT / "pddl" / "base_problem.pddl",
    projection_checkpoint=ROOT / "models" / "image_projection.pth",
)

print(plan)
