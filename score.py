import numpy as np
from scipy.optimize import linear_sum_assignment

def build_score_matrix(members, roles, history_df, new_cover_requests):
    scores = np.zeros((len(members), len(roles)))

    member_idx = {m: i for i, m in enumerate(members)}
    role_idx = {r: i for i, r in enumerate(roles)}

    # Build preference lookup
    pref_dict = {}
    for req in new_cover_requests:
        pref_dict[req["member_name"]] = {
            req.get("top_1_choice"): 1,
            req.get("top_2_choice"): 2,
            req.get("top_3_choice"): 3,
        }

    for member in members:
        m_data = history_df[history_df["member_name"] == member]

        got_1 = int(m_data["total_got_1"].max()) if not m_data.empty else 0
        got_2 = int(m_data["total_got_2"].max()) if not m_data.empty else 0
        got_3 = int(m_data["total_got_3"].max()) if not m_data.empty else 0
        got_none = int(m_data["total_got_none"].max()) if not m_data.empty else 0

        for role in roles:
            pref_rank = pref_dict.get(member, {}).get(role)

            if pref_rank is None:
                score = 0
            else:
                base = 4 - pref_rank
                penalty = got_1 + got_2 * 0.5 + got_3 * 0.25 + got_none * 0.1
                score = base / (1 + penalty)

            scores[member_idx[member], role_idx[role]] = -score

    return scores


def assign_roles(members, roles, history_df, new_cover_requests):
    scores = build_score_matrix(members, roles, history_df, new_cover_requests)
    row_ind, col_ind = linear_sum_assignment(scores)

    assignments = []
    for r, c in zip(row_ind, col_ind):
        assignments.append({
            "member": members[r],
            "assigned_role": roles[c],
            "score": -scores[r, c],
        })

    return assignments
