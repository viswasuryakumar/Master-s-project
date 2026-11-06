from pathlib import Path

from bgl_demo import analyze_text


sample_path = Path('data/BGL/raw-excerpt.log')
records, grouped = analyze_text(sample_path.read_text(encoding='latin-1'), window_size=2, step_size=2, threshold=1)

print('\n=== Parsed BGL records ===')
print(records[['Label', 'Id', 'Date', 'Time', 'Level', 'Message', 'Anomaly']].to_string(index=False))
print('\n=== Labels and sequence groups ===')
print(records[['Id', 'Label', 'Anomaly', 'Split']].to_string(index=False))
print(grouped[['Split', 'Sequence anomaly', 'Record labels', 'Sequence']].to_string(index=False))
print('\n=== Training records ===')
print(records.loc[records['Split'].eq('train'), ['Id', 'Label', 'Message']].to_string(index=False))
print('\n=== Testing records ===')
print(records.loc[records['Split'].eq('test'), ['Id', 'Label', 'Message']].to_string(index=False))
print('\n=== Training sequences ===')
print(grouped.loc[grouped['Split'].eq('train'), ['Sequence anomaly', 'Sequence']].to_string(index=False))
print('\n=== Testing sequences ===')
print(grouped.loc[grouped['Split'].eq('test'), ['Sequence anomaly', 'Sequence']].to_string(index=False))
