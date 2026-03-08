import re
import pandas as pd

SF1_PATH = "Data/SFdata.govdata.csv"
SF2_PATH = "Data/SFgov.orgdata.csv"

MERGED_OUTPUT = "SF_merged_data_with_month_ucr.csv"
MULTIHOT_OUTPUT = "SF_merged_data_multihot_with_month_ucr.csv"

# Columns
CANONICAL_COLUMNS = [
    "RecordID",
    "MonthOccurrence",
    "SuspectsRaceAsAGroup",
    "MostSeriousUCR",
    "MostSeriousLocation",
    "MostSeriousBias",
    "MostSeriousBiasType",
]

# Maps to rename features
SF1_RENAME_MAP = {
    "record_id": "RecordID",
    "occurence_month": "MonthOccurrence",
    "suspects_race_as_a_group": "SuspectsRaceAsAGroup",
    "most_serious_ucr": "MostSeriousUCR",
    "most_serious_location": "MostSeriousLocation",
    "most_serious_bias": "MostSeriousBias",
    "most_serious_bias_type": "MostSeriousBiasType",
}

SF2_RENAME_MAP = {
    "RecordId": "RecordID",
    "MonthOccurrence": "MonthOccurrence",
    "SuspectsRaceAsAGroup": "SuspectsRaceAsAGroup",
    "MostSeriousUcr": "MostSeriousUCR",
    "MostSeriousLocation": "MostSeriousLocation",
    "MostSeriousBias": "MostSeriousBias",
    "MostSeriousBiasType": "MostSeriousBiasType",
}

# Labels to ignore
IGNORE_LABELS = {"", "Nan", "None", "nan", "none"}

# protect phrases that contain "or" so they do not get split incorrectly
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

# how each column should be split for multi-hot encoding
DELIMS = {
    "SuspectsRaceAsAGroup": "or",
    "MostSeriousLocation": "/",
    "MostSeriousBias": "or",
    "MostSeriousBiasType": "/",
    "MostSeriousUCR": None,
    "MonthOccurrence": None,
}

MONTH_NAME_MAP = {
    1: "january",
    2: "february",
    3: "march",
    4: "april",
    5: "may",
    6: "june",
    7: "july",
    8: "august",
    9: "september",
    10: "october",
    11: "november",
    12: "december",
}


def load_and_standardize_sf1(path: str) -> pd.DataFrame:
    """
    Load SFdata.govdata.csv and rename/select columns to canonical labels.
    Also converts occurence_month to month number 1-12.
    """
    df = pd.read_csv(path)

    df = df.rename(columns=SF1_RENAME_MAP)
    df = df[list(SF1_RENAME_MAP.values())].copy()

    df["RecordID"] = df["RecordID"].astype("string").str.strip()

    # SF1 month values may be full dates like YYYY/MM/DD or similar
    month_dt = pd.to_datetime(df["MonthOccurrence"], errors="coerce")
    df["MonthOccurrence"] = month_dt.dt.month.astype("Int64")

    return df


def load_and_standardize_sf2(path: str) -> pd.DataFrame:
    """
    Load SFgov.orgdata.csv and rename/select columns to canonical labels.
    Keeps MonthOccurrence as numeric month if already stored that way.
    """
    df = pd.read_csv(path)

    df = df.rename(columns=SF2_RENAME_MAP)
    df = df[list(SF2_RENAME_MAP.values())].copy()

    df["RecordID"] = df["RecordID"].astype("string").str.strip()
    df["MonthOccurrence"] = pd.to_numeric(df["MonthOccurrence"], errors="coerce").astype("Int64")

    return df


def sanitize_for_colname(value: str) -> str:
    """
    Convert label text into a clean lowercase column suffix.
    Example:
    'Simple Assault' -> 'simple_assault'
    """
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def protect_or_phrases(text: str) -> str:
    """
    Temporarily replace 'or' inside phrases like
    'Black or African American' so split logic does not break them apart.
    """
    placeholder = "__OR__"
    out = text
    for phrase in PROTECTED_PHRASES:
        out = out.replace(phrase, phrase.replace(" or ", f" {placeholder} "))
    return out


def unprotect_or_placeholders(token: str) -> str:
    return token.replace("__OR__", "or")


def month_to_label(value) -> str | None:
    """
    Convert numeric month into lowercase month label.
    Example: 3 -> 'march'
    """
    if pd.isna(value):
        return None
    try:
        month_num = int(value)
    except Exception:
        return None
    return MONTH_NAME_MAP.get(month_num)


def split_cell(value, delim: str | None, column_name: str) -> list[str]:
    """
    Split one cell into one or more labels for multi-hot encoding.
    """
    if pd.isna(value):
        return []

    if column_name == "MonthOccurrence":
        month_label = month_to_label(value)
        return [month_label] if month_label else []

    s = str(value).strip()
    if not s:
        return []

    s = re.sub(r"\s+", " ", s)

    if delim == "/":
        tokens = [part.strip() for part in s.split("/") if part.strip()]

    elif delim == "or":
        protected = protect_or_phrases(s)
        parts = re.split(r"\s+or\s+", protected)
        tokens = [unprotect_or_placeholders(part.strip()) for part in parts if part.strip()]

    else:
        tokens = [s]

    cleaned = []
    for token in tokens:
        if token is None:
            continue
        t = str(token).strip()
        if t in IGNORE_LABELS:
            continue
        cleaned.append(t)

    return cleaned


def build_merged_dataframe(sf1_path: str, sf2_path: str) -> pd.DataFrame:
    """
    Preprocess both SF files so they have the same column names,
    concatenate them, and remove duplicate RecordID rows.
    """
    df1 = load_and_standardize_sf1(sf1_path)
    df2 = load_and_standardize_sf2(sf2_path)

    combined = pd.concat([df1, df2], ignore_index=True)

    # remove duplicate RecordIDs, keeping first occurrence
    combined = combined.drop_duplicates(subset=["RecordID"], keep="first").reset_index(drop=True)

    # force final column order
    combined = combined[CANONICAL_COLUMNS]

    return combined


def build_multihot_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Multi-hot encode the selected categorical columns.
    """
    out = pd.DataFrame({
        "RecordID": df["RecordID"].astype("string").str.strip()
    })

    for col, delim in DELIMS.items():
        label_lists = []
        all_labels = set()

        for value in df[col]:
            labels = split_cell(value, delim, col)
            label_lists.append(labels)
            all_labels.update(labels)

        # create one binary column per unique label
        for label in sorted(all_labels):
            new_col = f"{col}__{sanitize_for_colname(label)}"
            out[new_col] = 0

        # fill 1s where the row has that label
        for row_idx, labels in enumerate(label_lists):
            for label in labels:
                new_col = f"{col}__{sanitize_for_colname(label)}"
                out.at[row_idx, new_col] = 1

    return out


def main():
    # Step 1: preprocess + merge
    merged_df = build_merged_dataframe(SF1_PATH, SF2_PATH)
    merged_df.to_csv(MERGED_OUTPUT, index=False)

    # Step 2: multi-hot encode
    multihot_df = build_multihot_dataframe(merged_df)
    multihot_df.to_csv(MULTIHOT_OUTPUT, index=False)

    print("Saved merged file:", MERGED_OUTPUT, merged_df.shape)
    print("Saved multihot file:", MULTIHOT_OUTPUT, multihot_df.shape)


if __name__ == "__main__":
    main()