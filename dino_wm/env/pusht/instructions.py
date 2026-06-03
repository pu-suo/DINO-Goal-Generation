"""
Templated instruction generator for multi-color PushT.

Each instruction names the *active* target color and is the only signal that
disambiguates which of the N visible targets is the goal (text is load-bearing).
Templates carry stable integer ids so a subset can be held out later to test
paraphrase generalization (Phase 1+), independently of the color-location split.

`{color}` is the only slot. Keep phrasings natural and varied (synonyms for
"push"/"T"/"target", reordered clauses).
"""

# (id, template) — ids are STABLE; append new ones, never renumber.
INSTRUCTION_TEMPLATES = [
    (0, "push the T to the {color} target"),
    (1, "move the T onto the {color} target"),
    (2, "push the block to the {color} marker"),
    (3, "bring the T to the {color} goal"),
    (4, "navigate the T block to the {color} target"),
    (5, "slide the block onto the {color} region"),
    (6, "to the {color} target, push the T"),
    (7, "align the T with the {color} target"),
    (8, "get the block to the {color} target"),
    (9, "push the T onto the {color} one"),
]

# Default split for later paraphrase-generalization (NOT the headline split).
DEFAULT_HELDOUT_TEMPLATE_IDS = (7, 8, 9)


def template_ids(held_out=False, heldout_ids=DEFAULT_HELDOUT_TEMPLATE_IDS):
    """Return the list of template ids in the train (or held-out) pool."""
    heldout = set(heldout_ids)
    return [
        tid for tid, _ in INSTRUCTION_TEMPLATES
        if (tid in heldout) == bool(held_out)
    ]


def render_instruction(color, template_id):
    """Format the given template id with the color name."""
    tmpl = dict(INSTRUCTION_TEMPLATES)[template_id]
    return tmpl.format(color=color)


def sample_instruction(color, rng, held_out=False,
                       heldout_ids=DEFAULT_HELDOUT_TEMPLATE_IDS):
    """Sample (instruction_string, template_id) for a color from the chosen pool.

    rng: np.random.RandomState / Generator (must expose .choice).
    """
    pool = template_ids(held_out=held_out, heldout_ids=heldout_ids)
    template_id = int(rng.choice(pool))
    return render_instruction(color, template_id), template_id
