import pandas as pd
from io import StringIO
from score import assign_roles
from ml import train_model, predict_assignments

def load_history(filepath="data/history.csv"):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.strip().startswith("#")]
    return pd.read_csv(StringIO("".join(lines)))

def get_user_input():
    num_members = int(input("How many members are in this cover? "))
    requests = []

    for i in range(num_members):
        print(f"\nEnter info for member {i + 1}:")
        name = input("  Member name: ").strip().lower()
        top_1 = input("  Top 1 choice: ").strip().lower() or None
        top_2 = input("  Top 2 choice: ").strip().lower() or None
        top_3 = input("  Top 3 choice: ").strip().lower() or None

        requests.append({
            "member_name": name,
            "top_1_choice": top_1,
            "top_2_choice": top_2,
            "top_3_choice": top_3,
        })

    return requests

def main():
    history_df = load_history("data/history.csv")
    new_cover_requests = get_user_input()

    members = [req["member_name"] for req in new_cover_requests]

    raw_roles = input(
        "\nEnter all roles to assign (comma separated), or leave blank to auto-detect: "
    ).strip().lower()

    if raw_roles:
        roles = [r.strip() for r in raw_roles.split(",") if r.strip()]
    else:
        roles = list({
            r for req in new_cover_requests
            for r in (
                req.get("top_1_choice"),
                req.get("top_2_choice"),
                req.get("top_3_choice"),
            )
            if r
        })

    if len(roles) != len(members):
        print(
            f"\nWarning: Number of roles ({len(roles)}) "
            f"does not match number of members ({len(members)})."
        )
        confirm = input("Do you want to continue anyway? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return

    method = input("\nChoose assignment method ('score' or 'ml'): ").strip().lower()

    # using score matrix
    if method == "score":
        assignments = assign_roles(
            members=members,
            roles=roles,
            history_df=history_df,
            new_cover_requests=new_cover_requests,
        )
    # using machine learning
    elif method == "ml":
        ml_history_df = history_df
        new_data = []

        for req in new_cover_requests:
            member = req["member_name"]

            member_hist = ml_history_df[ml_history_df["member_name"] == member]

            past_covers = int(member_hist["total_covers"].max()) if not member_hist.empty else 0
            past_top_1 = int(member_hist["total_got_1"].max()) if not member_hist.empty else 0
            past_top_2 = int(member_hist["total_got_2"].max()) if not member_hist.empty else 0
            past_top_3 = int(member_hist["total_got_3"].max()) if not member_hist.empty else 0
            past_got_none = int(member_hist["total_got_none"].max()) if not member_hist.empty else 0

            for i, role in enumerate(
                [req.get("top_1_choice"), req.get("top_2_choice"), req.get("top_3_choice")],
                start=1
            ):
                if role:
                    new_data.append({
                        "member": member,
                        "role": role,
                        "preference_rank": i,
                        "total_covers": past_covers,
                        "total_got_1": past_top_1,
                        "total_got_2": past_top_2,
                        "total_got_3": past_top_3,
                        "total_got_none": past_got_none,
                    })

        model = train_model(ml_history_df)
        prediction_df = predict_assignments(model, new_data)

        assignments = []
        assigned_roles = set()

        for member in members:
            for _, row in prediction_df.iterrows():
                if row["member"] == member and row["role"] not in assigned_roles:
                    assignments.append({
                        "member": member,
                        "assigned_role": row["role"],
                        "score": row["prob"],
                    })
                    assigned_roles.add(row["role"])
                    break

    else:
        print("Invalid method. Choose 'score' or 'ml'.")
        return

    # output
    print("\nAssignments:")
    for a in assignments:
        print(f"{a['member']} → {a['assigned_role']} (score: {a['score']:.2f})")


if __name__ == "__main__":
    main()
