import json
import os
import datetime as dt

nb_path = "/Users/birdblues/workspace/fedwatch/test3.ipynb"

# Helper to read file
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Update fetch_macro_state_daily
cell_idx = -1
for i, cell in enumerate(nb["cells"]):
    # Use a robust check
    if cell["cell_type"] == "code" and "def fetch_macro_state_daily" in "".join(cell["source"]):
        cell_idx = i
        break

if cell_idx != -1:
    new_source = [
        "def fetch_macro_state_daily(start_date: dt.date, end_date: dt.date) -> pd.DataFrame:\n",
        "    endpoint = f\"{SUPABASE_URL}/rest/v1/macro_state_daily\"\n",
        "    headers = {\n",
        "        \"apikey\": SUPABASE_KEY,\n",
        "        \"Authorization\": f\"Bearer {SUPABASE_KEY}\",\n",
        "    }\n",
        "    params_base = {\n",
        "        \"select\": \",\".join(\n",
        "            [\n",
        "                \"date\",\n",
        "                \"regime_id\",\n",
        "                \"regime_label\",\n",
        "                \"stress_score\",\n",
        "                \"hmm_score_raw\",\n",
        "                \"hmm_score_0_100\",\n",
        "                \"prob_state_0\",\n",
        "                \"prob_state_1\",\n",
        "                \"prob_state_2\",\n",
        "                \"prob_state_3\",\n",
        "                \"skew\",\n",
        "                \"vix\",\n",
        "                \"sofr_minus_iorb\",\n",
        "            ]\n",
        "        ),\n",
        "        \"date\": [f\"gte.{start_date.isoformat()}\", f\"lte.{end_date.isoformat()}\"],\n",
        "        \"order\": \"date.asc\",\n",
        "        \"limit\": 1000,\n",
        "    }\n",
        "\n",
        "    rows = []\n",
        "    offset = 0\n",
        "    while True:\n",
        "        params = params_base.copy()\n",
        "        params[\"offset\"] = offset\n",
        "        resp = requests.get(endpoint, headers=headers, params=params, timeout=30)\n",
        "        resp.raise_for_status()\n",
        "        batch = resp.json()\n",
        "        rows.extend(batch)\n",
        "        if len(batch) < 1000:\n",
        "            break\n",
        "        offset += 1000\n",
        "\n",
        "    df = pd.DataFrame(rows)\n",
        "    if df.empty:\n",
        "        return df\n",
        "    df[\"date\"] = pd.to_datetime(df[\"date\"])\n",
        "    \n",
        "    # Convert numeric columns\n",
        "    cols = [\"stress_score\", \"hmm_score_raw\", \"hmm_score_0_100\", \"skew\", \"sofr_minus_iorb\"]\n",
        "    for c in cols:\n",
        "        if c in df.columns:\n",
        "            df[c] = pd.to_numeric(df[c], errors=\"coerce\")\n",
        "            \n",
        "    return df"
    ]
    nb["cells"][cell_idx]["source"] = new_source
    print(f"Updated fetch function in cell {cell_idx}")

# 2. Append/Update Plotting Cell
plot_source = [
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Ensure aligned data\n",
    "if \"hmm_score_raw\" not in df.columns:\n",
    "    print(\"Warning: New columns not found in df. Please re-run the fetch and join cells.\")\n",
    "else:\n",
    "    # Calculate SPX Cumulative Return %\n",
    "    df[\"spx_pct\"] = (df[\"spx_norm\"] / df[\"spx_norm\"].iloc[0] - 1) * 100\n",
    "\n",
    "    # Calculate Z-Scores for indicators (excluding Skew per request, excluding SPX)\n",
    "    cols_to_zscore = [\"vix\", \"sofr_minus_iorb\", \"hmm_score_raw\", \"stress_score\"]\n",
    "    for col in cols_to_zscore:\n",
    "        if col in df.columns:\n",
    "            df[f\"{col}_z\"] = (df[col] - df[col].mean()) / df[col].std()\n",
    "\n",
    "    # Calculate Skew Daily Percentage Change\n",
    "    if \"skew\" in df.columns:\n",
    "        df[\"skew_daily_pct\"] = df[\"skew\"].pct_change() * 100\n",
    "\n",
    "    fig, ax1 = plt.subplots(figsize=(14, 8))\n",
    "\n",
    "    # Ax1: SPX (%)\n",
    "    ax1.plot(df.index, df[\"spx_pct\"], label=\"SPX (Cumulative %)\", color=\"black\", linewidth=2, alpha=0.8)\n",
    "    ax1.set_ylabel(\"SPX Return (%)\")\n",
    "    ax1.legend(loc=\"upper left\")\n",
    "    ax1.grid(True, alpha=0.3)\n",
    "\n",
    "    # Ax2: Z-scores and Skew Daily Pct\n",
    "    ax2 = ax1.twinx()\n",
    "    \n",
    "    # VIX Z-score\n",
    "    if \"vix_z\" in df.columns:\n",
    "        ax2.plot(df.index, df[\"vix_z\"], label=\"VIX (Z-score)\", color=\"purple\", linewidth=1, alpha=0.6)\n",
    "\n",
    "    # SOFR-IORB Z-score\n",
    "    if \"sofr_minus_iorb_z\" in df.columns:\n",
    "        ax2.plot(df.index, df[\"sofr_minus_iorb_z\"], label=\"SOFR-IORB (Z-score)\", color=\"red\", linewidth=1, alpha=0.6)\n",
    "\n",
    "    # HMM Score Z-score\n",
    "    if \"hmm_score_raw_z\" in df.columns:\n",
    "        ax2.fill_between(df.index, df[\"hmm_score_raw_z\"], label=\"HMM Score (Z-score)\", color=\"green\", alpha=0.1)\n",
    "        ax2.plot(df.index, df[\"hmm_score_raw_z\"], color=\"green\", alpha=0.4)\n",
    "\n",
    "    # Stress Score Z-score\n",
    "    if \"stress_score_z\" in df.columns:\n",
    "        ax2.plot(df.index, df[\"stress_score_z\"], label=\"Stress Score (Z-score)\", color=\"orange\", linewidth=1.5, linestyle=\"-.\")\n",
    "\n",
    "    # Skew Daily Percentage Change\n",
    "    if \"skew_daily_pct\" in df.columns:\n",
    "        # Use a lighter line as daily percent can be noisy\n",
    "        ax2.plot(df.index, df[\"skew_daily_pct\"], label=\"Skew (Daily %)\", color=\"blue\", linewidth=0.8, alpha=0.5, linestyle=\"--\")\n",
    "\n",
    "    # Zero line for Z-scores/Pct\n",
    "    ax2.axhline(0, color=\"gray\", linestyle=\":\", alpha=0.8)\n",
    "    ax2.set_ylabel(\"Z-Score (Std Devs) / Skew Daily %\")\n",
    "    ax2.legend(loc=\"upper right\")\n",
    "\n",
    "    plt.title(\"Macro State Indicators (Z-Scores & Skew%) vs SPX\")\n",
    "    plt.tight_layout()\n",
    "    plt.show()"
]

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": plot_source
}

nb["cells"].append(new_cell)
print("Appended plotting cell")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=4)
