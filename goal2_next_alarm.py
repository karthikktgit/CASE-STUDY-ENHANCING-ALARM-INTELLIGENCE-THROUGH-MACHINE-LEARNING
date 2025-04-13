import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

# ------------------ Load and Create Transitions ------------------

# Load test data from Excel
xls = pd.ExcelFile("IM009B-XLS-ENG.xlsx")
test_data = pd.read_excel(xls, sheet_name="Test Data for LR and DT")

# Sort and create transitions
test_data_sorted = test_data.sort_values(by="S NO").reset_index(drop=True)
transitions = []

for i in range(len(test_data_sorted) - 1):
    current_tag = test_data_sorted.loc[i, "SO"]
    next_tag = test_data_sorted.loc[i + 1, "SO"]
    transitions.append((current_tag, next_tag))

df_transitions = pd.DataFrame(transitions, columns=["Current_Alarm", "Next_Alarm"])
df_transitions.to_csv("goal2_alarm_transitions.csv", index=False)

# ------------------ Build Markov Chain ------------------

transition_counts = df_transitions.groupby(["Current_Alarm", "Next_Alarm"]).size().unstack(fill_value=0)
transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)
transition_probs.to_csv("goal2_markov_chain_matrix.csv")

print("✅ Done: Transition matrix created and saved.")
print(transition_probs.head())

# ------------------ Predict Next Alarm ------------------

def predict_next_alarm(current_tag):
    if current_tag not in transition_probs.index:
        print(f"⚠️ '{current_tag}' not found in the Markov model.")
        return None
    next_alarm = transition_probs.loc[current_tag].idxmax()
    probability = transition_probs.loc[current_tag].max()
    print(f"\n🔮 Predicted next alarm after '{current_tag}': {next_alarm} (Probability: {probability:.2f})")
    return next_alarm

predict_next_alarm("XASS-3140")

# ------------------ Export Prediction Summary ------------------

prediction_summary = []
for current_tag in transition_probs.index:
    next_tag = transition_probs.loc[current_tag].idxmax()
    probability = transition_probs.loc[current_tag].max()
    prediction_summary.append({
        "Current Alarm": current_tag,
        "Most Likely Next Alarm": next_tag,
        "Probability": round(probability, 4)
    })

df_predictions = pd.DataFrame(prediction_summary)
df_predictions.to_csv("goal2_next_alarm_predictions.csv", index=False)

print("✅ Exported prediction summary to 'goal2_next_alarm_predictions.csv'")
print(df_predictions.head())

# ------------------ Filter & Save Confident Predictions ------------------

df_filtered = df_predictions[
    (df_predictions["Current Alarm"] != df_predictions["Most Likely Next Alarm"]) &
    (df_predictions["Probability"] > 0.3)
]
df_filtered.to_csv("goal2_predictions_filtered.csv", index=False)

print("✅ Saved: goal2_predictions_filtered.csv")
print(df_filtered.head())

# ------------------ Plot and Save Heatmap ------------------

top_alarms = transition_probs.sum(axis=1).sort_values(ascending=False).head(15).index
filtered_matrix = transition_probs.loc[top_alarms, top_alarms]

plt.figure(figsize=(12, 8))
sns.heatmap(filtered_matrix, annot=True, fmt=".2f", cmap="YlGnBu", cbar=True)
plt.title("🔥 Alarm-to-Alarm Transition Probabilities (Top 15 Tags)")
plt.xlabel("Next Alarm")
plt.ylabel("Current Alarm")
plt.tight_layout()
plt.savefig("goal2_transition_heatmap_local.png")
plt.close()

# ------------------ Plot and Save Bar Plot ------------------

top_predictions = df_filtered.sort_values(by="Probability", ascending=False).head(20)

plt.figure(figsize=(12, 8))
sns.barplot(
    y="Current Alarm",
    x="Probability",
    hue="Most Likely Next Alarm",
    data=top_predictions,
    dodge=False
)
plt.title("🔮 Top 20 Most Likely Alarm Transitions")
plt.xlabel("Transition Probability")
plt.ylabel("Current Alarm")
plt.legend(title="Next Alarm", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.savefig("goal2_top20_transitions_barplot_local.png")
plt.close()

# ------------------ Display Heatmap Only ------------------
from PIL import Image

heatmap_img = Image.open("goal2_transition_heatmap_local.png")
plt.figure(figsize=(10, 6))
plt.imshow(heatmap_img)
plt.axis('off')
plt.title("🔥 Alarm Transition Heatmap", fontsize=14)
plt.tight_layout()
plt.show()

# ------------------ Display Barplot Only ------------------

barplot_img = Image.open("goal2_top20_transitions_barplot_local.png")
plt.figure(figsize=(10, 6))
plt.imshow(barplot_img)
plt.axis('off')
plt.title("🔮 Top 20 Alarm Transitions", fontsize=14)
plt.tight_layout()
plt.show()
