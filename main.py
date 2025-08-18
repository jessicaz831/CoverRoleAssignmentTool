import pandas as pd
from scipy.optimize import linear_sum_assignment
import numpy as np
import csv
from io import StringIO

def load_history(filepath="data/history.csv"):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [line for line in f if not line.strip().startswith("#")]
    return pd.read_csv(StringIO("".join(lines)))

def build_score_matrix(members, roles, history_df, new_cover_requests):
    scores = np.zeros((len(members), len(roles)))
    member_idx = {m: i for i, m in enumerate(members)}
    role_idx = {r: i for i, r in enumerate(roles)}

    pref_dict = {}
    for req in new_cover_requests:
        pref_dict[req['member_name']] = {
            req.get('top_1_choice') or "": 1,
            req.get('top_2_choice') or "": 2,
            req.get('top_3_choice') or "": 3
        }

    for m in members:
        m_data = history_df[history_df['member_name'] == m]
        got_1 = m_data['total_got_1'].max() if not m_data.empty else 0
        got_2 = m_data['total_got_2'].max() if not m_data.empty else 0
        got_3 = m_data['total_got_3'].max() if not m_data.empty else 0
        got_none = m_data['total_got_none'].max() if not m_data.empty else 0

        for r in roles:
            pref_rank = pref_dict.get(m, {}).get(r, None)
            if pref_rank is None:
                score = 0
            else:
                base = 4 - pref_rank
                penalty = got_1 + got_2 * 0.5 + got_3 * 0.25 + got_none * 0.1
                score = base / (1 + penalty)
            scores[member_idx[m], role_idx[r]] = -score

    return scores, member_idx, role_idx

# hungarian algorithm
def assign_roles(members, roles, history_df, new_cover_requests):
    scores, member_idx, role_idx = build_score_matrix(members, roles, history_df, new_cover_requests)
    row_ind, col_ind = linear_sum_assignment(scores)
    assignments = []
    for r, c in zip(row_ind, col_ind):
        assignments.append({
            'member': members[r],
            'assigned_role': roles[c],
            'score': -scores[r, c]
        })
    return assignments

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

    members = [req['member_name'] for req in new_cover_requests]

    raw_roles = input("\nEnter all roles to assign (comma separated), or leave blank to auto-detect: ").strip().lower()
    if raw_roles:
        roles = [r.strip() for r in raw_roles.split(",") if r.strip()]
    else:
        # detect roles from lists if left blank
        roles = list({r for req in new_cover_requests for r in [
            req.get('top_1_choice'), req.get('top_2_choice'), req.get('top_3_choice')] if r})

    # check if role count matches member count
    if len(roles) != len(members):
        print(f"\nWarning: Number of roles ({len(roles)}) does not match number of members ({len(members)}).")
        confirm = input("Do you want to continue anyway? (yes/no): ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            return

    assignments = assign_roles(members, roles, history_df, new_cover_requests)

    print(f"\nAssignments:")
    for a in assignments:
        print(f"{a['member']} → {a['assigned_role']} (score: {a['score']:.2f})")

if __name__ == "__main__":
    main()
