# Neuro-Symbolic Planning

This project connects image recognition with symbolic planning. The input can be either the name of an object or an image from CIFAR-100. The system identifies the object, represents it as an object inside a planning environment, and searches for a sequence of actions that reaches a requested goal.

The two parts of the system do different jobs. The neural part deals with recognition. It turns an image into a numerical representation and compares it with numerical representations of object names. The symbolic part deals with actions. It works with explicit facts such as `apple is in the lab`, `the agent is holding the knife`, or `the tiger has been photographed`.

A plan is simply the ordered list of actions that changes the starting facts into the goal facts.

## A simple example

Suppose the object is an apple. The starting state says that the apple is in the lab and is whole. The goal says that the apple should be cut into pieces.

```python
plan = generate_plan(
    "apple",
    initial_state=["(at ?x lab)", "(whole ?x)"],
    goal_state=["(cut-into-pieces ?x)"],
    domain_file=DOMAIN_FILE,
    base_problem=PROBLEM_FILE,
    projection_checkpoint=MODEL_FILE,
)
```

The planner returns:

```text
(pick-up knife lab)
(slice-object apple lab)
```

The result follows from the rules of the planning environment. The apple cannot be sliced unless the agent is in the same place as the apple, the apple is whole and clear, and the agent is holding the knife. The knife starts in the lab, so the planner first picks it up and then slices the apple.

The same planning code can receive the object name from the image model instead of directly from text.

## What the planning environment contains

The symbolic part of the project works inside a small environment described in PDDL. PDDL stands for Planning Domain Definition Language. It is a text format for describing objects, facts, actions, and goals so that a planner can reason about them.

The environment has two locations:

- `lab`
- `outdoors`

It also contains the 100 object categories from CIFAR-100. These include objects such as `apple`, `bottle`, `table`, `bus`, `tiger`, `train`, `rose`, and `lawn-mower`.

The initial setup places different objects in different parts of the environment. Furniture, food, small electronics, containers, and several small animals are in the lab. Vehicles, plants, structures, natural features, and many animals are outdoors.

The lab also contains two tools:

- a `knife`, which is needed to cut an object
- a `dslr`, which is needed to photograph an object

Some objects are marked as surfaces. These are objects that other objects can be placed on. The surfaces in the domain are `table`, `bed`, `chair`, `couch`, `road`, `bridge`, `plain`, `sea`, `plate`, and `bowl`.

The environment also keeps track of facts about the current situation. For example:

```text
(agent-at lab)
(at apple lab)
(whole apple)
(clear apple)
(hand-empty)
```

These statements mean that the agent is in the lab, the apple is in the lab, the apple has not been cut, nothing is stacked on top of the apple, and the agent is not holding anything.

As actions are performed, some of these facts are removed and others are added. This is how the planner keeps track of what has changed.

## What actions the planner can use

The planner has seven action types. Each action has conditions that must already be true before it can happen.

### Walk between the two locations

The agent can move between the lab and the outdoor area.

For example:

```text
(walk-between-rooms lab outdoors)
```

The agent must currently be at the starting location. After the action, the old location is removed from the state and the new location becomes the agent's current location.

### Pick up an object

For example:

```text
(pick-up apple lab)
```

To pick something up:

- the agent must be in the same location as the object
- the object must be at that location
- the agent's hand must be empty
- the object must be clear, meaning nothing is stacked on top of it

After the action, the object is no longer considered to be sitting at that location. The agent is now holding it and no longer has an empty hand.

### Put an object down

For example:

```text
(put-down apple lab)
```

The agent must be holding the object and must be at the location where it is being put down. The action returns the object to that location and makes the agent's hand empty again.

### Stack one object on another

For example:

```text
(stack apple table lab)
```

To stack an object:

- the agent must be holding the object that will go on top
- the lower object must be in the same location
- the lower object must be clear
- the lower object must be one of the objects marked as a surface
- an object cannot be stacked on itself

After stacking, the top object is recorded as being on the lower object. The agent's hand becomes empty and the lower object is no longer clear.

### Unstack an object

For example:

```text
(unstack apple table lab)
```

The top object must currently be on the lower object, the agent must be in the same location, the top object must be clear, and the agent's hand must be empty. After unstacking, the agent is holding the top object and the lower object becomes clear again.

### Slice an object

For example:

```text
(slice-object apple lab)
```

The agent can slice an object only when:

- the agent and the object are in the same location
- the agent is holding the knife
- the object is whole
- the object is clear
- the object is not itself a tool

After slicing, the fact `whole apple` is removed and `cut-into-pieces apple` is added.

### Take a photo of an object

For example:

```text
(take-photo tiger outdoors)
```

The agent must be in the same location as the object and must be holding the DSLR. The object must also be clear and cannot be one of the tools. After the action, the object is marked as documented.

## What a goal means

A goal is a fact, or a set of facts, that should be true when the plan is finished.

The user supplies goals using the same simple predicate format used for the state. The placeholder `?x` is replaced with the object identified from the text or image input.

For example:

```python
goal_state=["(cut-into-pieces ?x)"]
```

If the input is `apple`, the planner reads this as:

```text
(cut-into-pieces apple)
```

Other goals can ask the planner to photograph an object, hold it, place it on a surface, or make another state described by the available predicates.

Here are several examples produced by the current planner.

### Goal: cut an apple into pieces

Starting information:

```text
apple is in the lab
apple is whole
```

Goal:

```text
apple is cut into pieces
```

Plan:

```text
(pick-up knife lab)
(slice-object apple lab)
```

### Goal: photograph a tiger that is outdoors

Starting information:

```text
tiger is outdoors
tiger is whole
```

Goal:

```text
tiger is documented
```

The DSLR is in the lab, while the tiger is outdoors. The planner therefore has to travel to the camera, pick it up, return outdoors, and take the photo.

```text
(walk-between-rooms outdoors lab)
(pick-up dslr lab)
(walk-between-rooms lab outdoors)
(take-photo tiger outdoors)
```

### Goal: hold a cup

Starting information:

```text
cup is in the lab
```

Goal:

```text
the agent is holding the cup
```

Plan:

```text
(pick-up cup lab)
```

### Goal: place an apple on the table

Starting information:

```text
apple is in the lab
```

Goal:

```text
apple is on top of the table
```

Plan:

```text
(pick-up apple lab)
(stack apple table lab)
```

These examples are useful because the plan is not generated as free-form text. Every returned action has to satisfy the conditions defined in the domain, and every action changes the state in a defined way.

## How the full system works

The project has three main stages.

```text
image -----------------> image encoder --------------------+
                                                           |
text label ------------------------------------------------+
                                                           v
                                                  identified object
                                                           |
                                                           v
                                             insert object into the
                                                planning problem
                                                           |
                                                           v
                                                  A* plan search
                                                           |
                                                           v
                                                   action sequence
```

Text input skips the image-recognition stage. This is useful when testing the planner by itself because the object name is already known.

Image input runs through the complete pipeline. The model first predicts an object name. That predicted name is then used by the planner.

## Representing object names as numbers

The project includes a saved word-embedding model. A word embedding represents each word as a vector, which is a list of numbers. In this model, each word is represented by 128 numbers.

The saved vocabulary contains 523 words. Words that were learned in similar contexts can end up close to each other in the embedding space.

For example, the nearest words to `man` in the saved model are:

```text
woman
boy
person
child
lady
```

The nearest words to `car` are:

```text
tractor
pickup_truck
bus
train
truck
```

The important role of these embeddings in the full pipeline is that they give the image model a numerical target for each object name. The model does not directly output a class number such as class 0 or class 57. Instead, it learns to place an image close to the vector representing its class name.

## Turning an image into an object name

For image input, the project uses MobileNetV3-small as the image encoder. A projection layer then converts the image features into a 128-dimensional vector so that the image can be compared with the 128-dimensional word vectors.

CIFAR-100 contains 100 image categories. Examples include `apple`, `bus`, `tiger`, `table`, `tractor`, and `lawn_mower`.

For a new image, the system performs these steps:

1. Resize and normalize the image.
2. Pass it through the MobileNetV3-small encoder.
3. Project the image features into 128 dimensions.
4. Compare the resulting vector with the saved vectors for all 100 CIFAR-100 class names.
5. Choose the class name with the highest cosine similarity.

Cosine similarity measures how closely two vectors point in the same direction. Here it is used to decide which object-name vector is closest to the image vector.

The saved image checkpoint contains the model weights, the 100 class-name vectors, and training metadata. The checkpoint records:

- epoch: **25**
- validation loss: **5.0123**
- mean validation cosine similarity: **0.6391**

The checkpoint already contains the MobileNet weights, so running inference does not require downloading a second set of pretrained weights.

## From the predicted name to the planner

The image model and the PDDL files do not use exactly the same spelling for every class. CIFAR-100 labels use underscores in names such as `lawn_mower` and `aquarium_fish`, while the PDDL problem uses `lawn-mower` and `aquarium-fish`.

The pipeline normalizes these names before passing them to the planner. This is a small step, but without it the perception and planning parts would refer to different object names.

The predicates supplied to `generate_plan` use `?x` as a placeholder. Once the object has been identified, the placeholder is replaced with the real object name.

For example:

```text
(at ?x outdoors)
(documented ?x)
```

becomes this for a tiger:

```text
(at tiger outdoors)
(documented tiger)
```

The resulting facts are inserted into a temporary PDDL problem and passed to the planner.

## How the planner searches

The planner parses the action definitions and the current problem state, then creates concrete versions of the actions for the objects and locations in the environment.

For example, the general action definition `pick-up` can become concrete actions such as:

```text
(pick-up apple lab)
(pick-up knife lab)
(pick-up tiger outdoors)
```

The planner then uses A* search. Each search state is a set of facts describing the current situation. Applying an action removes facts that are no longer true and adds facts that become true.

A* keeps track of two things:

- how many actions have already been used
- a heuristic estimate of how far the current state is from the goal

The heuristic gives extra cost when a goal is still unsatisfied. It also recognises two useful tool requirements. If the goal is to cut an object and the knife is not being held, the state receives an additional penalty. If the goal is to document an object and the DSLR is not being held, it receives a similar penalty.

This does not tell the planner the complete solution. It only helps it explore states that are more likely to lead to the requested goal.

## Text input and image input

Text and image inputs enter the same planning system at different points.

With text input, the object is already known. If the input is `apple`, the planner can immediately ground `?x` as `apple` and begin searching.

With image input, the object first has to be predicted. If the model predicts `apple`, the same planning process then begins with `apple` as the object symbol.

This separation is useful when checking the system. The planner can be tested independently with known text labels. The image model can also be tested separately from planning.

It also makes an important limitation visible. The planner can return a completely valid plan for the wrong object if the image model misclassifies the image. Symbolic reasoning can check whether the actions follow the rules, but it cannot correct an incorrect object label that it receives from the perception model.

## What I learned from the project

One of the clearest lessons from this project was that prediction and planning have different meanings of correctness.

For the image model, the important question is whether the image is placed close to the correct object name in the embedding space. For the planner, the important question is whether each action is allowed in the current state and whether the final state contains the requested goal facts.

These two forms of correctness are connected, but they are not the same. If an image of a tiger is classified as a lion, the planner may still produce a valid sequence for a lion. The reasoning can be correct even though the input symbol is wrong.

Another lesson was how much the interface between components matters. A difference as small as `aquarium_fish` versus `aquarium-fish` can stop two otherwise working parts of the system from communicating. In the cleaned version of the project, this conversion happens explicitly in one place.

The symbolic part also made the reasoning process easier to inspect. If the goal is to photograph a tiger outdoors, I can see why the plan travels to the lab first. The camera is in the lab, and the `take-photo` action requires the agent to be holding it. Each step can be checked against a specific requirement in the domain rather than treated as an unexplained model output.

I also found it useful to see how a search heuristic can use information from the goal without hard-coding a solution. The planner does not contain a rule saying that every photography task must follow one fixed sequence. It simply gives preference to states that satisfy more of the goal and recognises when a required tool is missing. The actual action sequence still depends on the current state.

## Repository structure

```text
neuro-symbolic-planning/
├── src/neuro_symbolic/
│   ├── embeddings.py       # load and inspect the saved word vectors
│   ├── perception.py       # encode images and predict object names
│   ├── planning.py         # parse PDDL and search for plans with A*
│   └── pipeline.py         # connect object recognition to planning
├── pddl/
│   ├── domain.pddl         # action definitions and their requirements
│   └── base_problem.pddl   # objects and the initial environment
├── models/
│   ├── word_embeddings.pth
│   └── image_projection.pth
├── examples/
│   ├── embedding_neighbors.py
│   └── text_plan.py
├── tests/
│   └── test_pipeline.py
├── pyproject.toml
└── README.md
```

## Running the project

Python 3.10 or newer is recommended.

Install the package from the repository root:

```bash
pip install -e .
```

Run the text planning example:

```bash
python examples/text_plan.py
```

This runs the apple example shown above and prints the generated plan.

Inspect nearest neighbours in the saved word-embedding space:

```bash
python examples/embedding_neighbors.py
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

The text planning example does not require image data. Image prediction expects an RGB image represented as a PyTorch tensor. The `preprocess_image` function resizes the image to the size expected by MobileNet and applies the same normalization used by the model.

## Current limits

The perception model can only choose from the 100 CIFAR-100 class names. An image of an object outside those categories will still be compared with those 100 names, so the returned label should not be treated as an open-ended image recognition result.

The planning environment is deliberately limited. It has two locations, two tools, and seven action types. It is useful for showing how object recognition can feed into symbolic planning, but it is not a general robot simulator.

For image input, only the highest-scoring predicted class is sent to the planner. The planner does not currently reason over several possible labels or use the confidence of the image model when choosing actions.

The PDDL parser implements the subset of PDDL needed by this domain. It is not a complete parser for every PDDL feature.
