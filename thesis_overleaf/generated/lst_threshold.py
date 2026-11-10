def at_threshold(df, signal, tau):
    """Metrics if we withhold every answer whose confidence is below tau."""
    keep = (df[signal] >= tau) & ~df["model_abstained"]
    n = len(df)
    correct = int((df["correct"] & keep).sum())
    halluc = int((df["hallucinated"] & keep).sum())
    answered = int(keep.sum())
    return {
        "threshold": tau, "coverage": answered / n,
        "accuracy": correct / n,                          # abstaining counts as wrong
        "selective_accuracy": correct / answered if answered else float("nan"),
        "hallucination_rate": halluc / answered if answered else 0.0,
        "hallucinations": halluc, "correct": correct,
        "score": (correct - halluc) / n,
    }

def choose_threshold(df, signal):
    best = max((at_threshold(df, signal, t) for t in thresholds_for(df[signal])),
               key=lambda m: (m["score"], m["coverage"]))
    return best["threshold"]
