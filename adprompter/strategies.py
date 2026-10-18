"""Persuasion strategies the agent chooses between. Tuple order is the action index, so
it must stay stable once a model has been trained against it."""

STRATEGIES: tuple[str, ...] = (
    "Liking", "Social Proof", "Scarcity", "Authority", "Consistency", "Reciprocity",
    "Contrast", "Unity", "Consensus", "Emotion", "Urgency", "Exclusivity", "Curiosity",
    "Anticipation", "Mystery", "Surprise", "Humor",
)

# One visual cue per strategy; the prompt builder turns it into image direction.
CUES: dict[str, str] = {
    "Liking": "warm, friendly faces enjoying the product together",
    "Social Proof": "a crowd of delighted customers, everyone holding one",
    "Scarcity": "the last unit on an otherwise empty shelf, spotlight",
    "Authority": "an expert in a lab coat endorsing it, clean studio",
    "Consistency": "a daily ritual, same product every morning, calm routine",
    "Reciprocity": "a gift box opening, a hand offering it forward",
    "Contrast": "split frame: dull before on the left, vivid after on the right",
    "Unity": "a diverse community sharing it around one table",
    "Consensus": "five-star reviews floating around the product",
    "Emotion": "a tender, cinematic moment, soft golden light",
    "Urgency": "a countdown clock, bold motion blur, act-now energy",
    "Exclusivity": "velvet rope, members only, black and gold palette",
    "Curiosity": "the product half-revealed behind a curtain",
    "Anticipation": "unboxing frozen mid-motion, eyes wide",
    "Mystery": "fog, silhouette, a single beam on the product",
    "Surprise": "confetti burst, an unexpected reveal",
    "Humor": "a playful, absurd scene, cartoon exaggeration",
}

if len(STRATEGIES) != len(set(STRATEGIES)) or set(STRATEGIES) != set(CUES):
    raise ValueError("strategies must be unique and each must have exactly one cue")
