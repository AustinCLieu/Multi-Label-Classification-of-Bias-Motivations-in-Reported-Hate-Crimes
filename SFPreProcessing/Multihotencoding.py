import re
import pandas as pd

INPUT_CSV = "Data/SF_merged_data.csv"
OUTPUT_CSV = "Data/SF_merged_data_multihot.csv"
ID_COL = "RecordID"

# Columns and delimiters
DELIMS = {
    "SuspectsRaceAsAGroup": "or",
    "MostSeriousLocation": "/",
    "MostSeriousBias": "or",
    "MostSeriousBiasType": "/"
}

# If True, prevents splitting labels like "Black or African American"
PROTECT_OR_PHRASES = True

# Common phrases where "or" is part of ONE label, not a separator between labels
PROTECTED_PHRASES = [
    "Black or African American",
    "American Indian or Alaska Native",
    "Native Hawaiian or Other Pacific Islander",
    "Hispanic or Latino",
    "Anti-Black or African American",
    "Anti-American Indian or Alaska Native",
    "Anti-Native Hawaiian or Other Pacific Islander",
    "Anti-Hispanic or Latino",
]


IGNORE_LABELS = set([
    "", "Nan", "None"
])

# Helpers

def sanitize_for_colname(s: str) -> str:
    # Turn a label into a safe column suffix
    # For example, "Race/Ethnicity/Ancestry" -> "race_ethnicity_ancestry"
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def protect_or_phrases(text: str) -> str:
    # Replace "or" inside protected phrases with a placeholder so it won't split
    if not PROTECT_OR_PHRASES:
        return text
    placeholder = "__OR__"
    out = text
    for phrase in PROTECTED_PHRASES:
        # Replace the exact phrase's "or" with placeholder version
        # For example "Black or African American" -> "Black__OR__African American"
        protected_version = phrase.replace(" or ", f" {placeholder} ")
        out = out.replace(phrase, protected_version)
    return out

def unprotect_or_placeholders(token: str) -> str:
    return token.replace("__OR__", "or")

def split_cell(value, delim: str) -> list[str]:
    """
    Convert a single cell into a list of labels based on the delimiter.
    """
    if pd.isna(value):
        return []

    s = str(value).strip()
    if s == "":
        return []

    # Normalize weird whitespace
    s = re.sub(r"\s+", " ", s)

    tokens = []

    if delim == "/":
        parts = s.split("/")
        for p in parts:
            t = p.strip()
            if t != "":
                tokens.append(t)

    elif delim == "or":
        # Protect phrases like "Black or African American" if enabled
        s2 = protect_or_phrases(s)

        # Split on the word "or" as a separator
        # For example, " A or B " -> ["A", "B"]
        # This uses spaces around or so it doesn't split "Organization"
        parts = re.split(r"\s+or\s+", s2)

        for p in parts:
            t = p.strip()
            if t != "":
                t = unprotect_or_placeholders(t)
                tokens.append(t)

    else:
        # Fallback, no split
        tokens = [s]

    # Remove ignored labels
    cleaned = []
    for t in tokens:
        if t is None:
            continue
        tt = str(t).strip()
        if tt == "":
            continue
        # normalize case for ignore check
        if tt in IGNORE_LABELS:
            continue
        cleaned.append(tt)

    return cleaned

# Main
def main():
    df = pd.read_csv(INPUT_CSV)

    if ID_COL not in df.columns:
        raise ValueError(f"Expected '{ID_COL}' column, but columns are: {list(df.columns)}")

    # Ensure RecordID is stable (string) and trimmed
    df[ID_COL] = df[ID_COL].astype("string").str.strip()

    # Output starts with RecordID
    out = pd.DataFrame({ID_COL: df[ID_COL]})

    # For each column we want to multi-hot encode
    for col, delim in DELIMS.items():
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not found in CSV. Skipping.")
            continue

        # Convert each row to a list of labels
        label_lists = []
        all_labels = set()

        for val in df[col]:
            labels = split_cell(val, delim)
            label_lists.append(labels)
            for lab in labels:
                all_labels.add(lab)

        # Create one column per unique label
        # Sort so column order is stable across runs
        sorted_labels = sorted(all_labels)

        # Initialize all zeros
        for lab in sorted_labels:
            new_col = f"{col}__{sanitize_for_colname(lab)}"
            out[new_col] = 0

        # Fill 1s where label is present
        for i, labels in enumerate(label_lists):
            for lab in labels:
                new_col = f"{col}__{sanitize_for_colname(lab)}"
                out.at[i, new_col] = 1

        print(f"[OK] Encoded '{col}' -> {len(sorted_labels)} columns (delimiter='{delim}')")

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}")
    print("Final shape:", out.shape)

if __name__ == "__main__":
    main()