#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 28 19:22:45 2025

@author: JessicaZhu
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

FEATURES = ["preference_rank", "past_covers", "past_top_choices", "past_backups", "role", "member"]

def load_history_data(filepath):
    return pd.read_csv(filepath)

def train_model(df):
    df = df.copy()
    df["role"] = LabelEncoder().fit_transform(df["role"])
    df["member"] = LabelEncoder().fit_transform(df["member"])

    X = df[FEATURES]
    y = df["assigned"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_assignments(model, new_data):
    df = pd.DataFrame(new_data).copy()
    df["role"] = LabelEncoder().fit_transform(df["role"])
    df["member"] = LabelEncoder().fit_transform(df["member"])

    df["prob"] = model.predict_proba(df[FEATURES])[:, 1]
    return df.sort_values(by="prob", ascending=False)
