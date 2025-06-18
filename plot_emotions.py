import pandas as pd
import matplotlib.pyplot as plt

# Load emotion logs
df = pd.read_csv("emotion_logs.csv")

# Count how many times each emotion occurred
emotion_counts = df['emotion'].value_counts()

# Plot as a bar chart
plt.figure(figsize=(10, 6))
emotion_counts.plot(kind='bar', color='skyblue')
plt.title("Emotion Frequency (Logged from Webcam)")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.grid(axis='y')
plt.tight_layout()
plt.show()
