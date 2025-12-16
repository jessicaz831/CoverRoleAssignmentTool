import pandas as pd
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "preference_rank",
    "total_covers",
    "total_got_1",
    "total_got_2",
    "total_got_3",
    "total_got_none",
]

def expand_history(history_df):
    """
    Convert each cover row into (member, role) training samples.
    """
    rows = []

    for _, r in history_df.iterrows():
        for rank, role in enumerate(
            [r["top_1_choice"], r["top_2_choice"], r["top_3_choice"]],
            start=1,
        ):
            if pd.isna(role):
                continue

            rows.append({
                "member": r["member_name"],
                "role": role,
                "preference_rank": rank,
                "total_covers": r["total_covers"],
                "total_got_1": r["total_got_1"],
                "total_got_2": r["total_got_2"],
                "total_got_3": r["total_got_3"],
                "total_got_none": r["total_got_none"],
                "assigned": int(role == r["assigned_role"]),
            })

    return pd.DataFrame(rows)


def train_model(history_df):
    """
    Train a Random Forest to predict whether a (member, role) pair is assigned.
    """
    train_df = expand_history(history_df)

    X = train_df[FEATURES]
    y = train_df["assigned"]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X, y)

    return model


def predict_assignments(model, new_data):
    """
    Predict probabilities for new (member, role) pairs.
    """
    df = pd.DataFrame(new_data)

    probs = model.predict_proba(df[FEATURES])[:, 1]

    df["prob"] = probs
    return df.sort_values("prob", ascending=False)
