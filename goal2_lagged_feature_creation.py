import pandas as pd

# Load your cleaned transitions dataset
df = pd.read_csv("goal2_alarm_transitions.csv")

# Extract just the sequence of alarms
alarm_sequence = df['Current_Alarm'].tolist()

# Create a mapping from alarm tag to integer
unique_tags = sorted(set(alarm_sequence))
tag_to_int = {tag: i for i, tag in enumerate(unique_tags)}
int_to_tag = {i: tag for tag, i in tag_to_int.items()}

# Convert alarm tags to integers
encoded_sequence = [tag_to_int[tag] for tag in alarm_sequence]

# Set how many lags (previous alarms) to use
n_lags = 3

# Create lagged dataset
X = []
y = []
for i in range(n_lags, len(encoded_sequence)):
    X.append(encoded_sequence[i - n_lags:i])  # previous 3 alarms
    y.append(encoded_sequence[i])             # next alarm

# Convert to DataFrame
X_df = pd.DataFrame(X, columns=[f"Lag_{j+1}" for j in range(n_lags)])
y_df = pd.DataFrame(y, columns=["Next_Alarm"])
final_df = pd.concat([X_df, y_df], axis=1)

# Map back integer codes to alarm tags for readability (optional)
final_df = final_df.applymap(lambda x: int_to_tag[x])

# Save to CSV
final_df.to_csv("goal2_lagged_alarm_sequences.csv", index=False)
print("✅ Saved lagged alarm sequence dataset: goal2_lagged_alarm_sequences.csv")
