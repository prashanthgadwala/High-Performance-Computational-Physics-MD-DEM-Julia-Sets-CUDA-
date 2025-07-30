import pandas as pd
import matplotlib.pyplot as plt

# Read your CSV file
df = pd.read_csv('performance.csv')

# Create a bar plot
plt.figure(figsize=(8,5))
bars = plt.bar(df['Method'] + ' (Cutoff: ' + df['Cutoff'] + ')', df['TimePerStep'], color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.ylabel('Average Time per Step (s)')
plt.title('Performance Comparison of MD Simulation')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('performance_comparison.png')
plt.show()