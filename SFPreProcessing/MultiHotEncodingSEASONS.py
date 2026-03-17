import pandas as pd
import re

df = pd.read_csv('CA_final_data_reduced.csv')

# ── Season mapping ───────────────────────────────────────────────────────────
# Winter: Dec(12), Jan(1), Feb(2)
# Spring: Mar(3), Apr(4), May(5)
# Summer: Jun(6), Jul(7), Aug(8)
# Fall:   Sep(9), Oct(10), Nov(11)
season_map = {
    1: 'winter', 2: 'winter', 12: 'winter',
    3: 'spring', 4: 'spring',  5: 'spring',
    6: 'summer', 7: 'summer',  8: 'summer',
    9: 'fall',  10: 'fall',   11: 'fall',
}
df['TimeOfYear'] = df['MonthOccurrence'].map(season_map)

# ── Normalise a string → snake_case column suffix ────────────────────────────
def normalise(s):
    s = str(s).lower().strip()
    s = re.sub(r'[^a-z0-9]+', '_', s)   # non-alphanumeric → _
    s = s.strip('_')
    return s

def multihot(df, col, prefix):
    dummies = pd.get_dummies(df[col], prefix=prefix, prefix_sep='__')
    dummies.columns = [f"{prefix}__{normalise(c.split('__', 1)[1])}" for c in dummies.columns]
    return dummies.astype(int)

# ── Encode each feature ──────────────────────────────────────────────────────
race    = multihot(df, 'SuspectsRaceAsAGroup', 'SuspectsRaceAsAGroup')
loc     = multihot(df, 'MostSeriousLocation',  'MostSeriousLocation')
bias    = multihot(df, 'MostSeriousBias',       'MostSeriousBias')
btype   = multihot(df, 'MostSeriousBiasType',   'MostSeriousBiasType')
ucr     = multihot(df, 'MostSeriousUCR',        'MostSeriousUCR')
season  = multihot(df, 'TimeOfYear',            'TimeOfYear')

# ── Assemble final dataframe ─────────────────────────────────────────────────
out = pd.concat([df[['RecordID']], race, loc, bias, btype, ucr, season], axis=1)

out.to_csv('CA_multihot_seasons.csv', index=False)
print("Saved! Shape:", out.shape)