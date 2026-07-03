import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# خوندن نتایج
df = pd.read_csv('cew_results.csv')

print("="*50)
print("📊 ANALYSIS RESULTS")
print("="*50)
print(f"📸 Total images processed: {len(df)}")
print(f"📈 Mean EAR: {df['ear'].mean():.4f}")
print(f"📉 Min EAR: {df['ear'].min():.4f}")
print(f"📈 Max EAR: {df['ear'].max():.4f}")
print(f"📊 Std EAR: {df['ear'].std():.4f}")

# دسته‌بندی: چشم باز (>0.25) و چشم بسته (<0.2)
open_eyes = df[df['ear'] > 0.25]
closed_eyes = df[df['ear'] < 0.2]
unknown = df[(df['ear'] >= 0.2) & (df['ear'] <= 0.25)]

print("\n📋 Classification:")
print(f"   👁️ Open eyes (>0.25): {len(open_eyes)} images")
print(f"   😴 Closed eyes (<0.2): {len(closed_eyes)} images")
print(f"   ❓ Unknown (0.2-0.25): {len(unknown)} images")

# رسم نمودار
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# هیستوگرام
ax1.hist(df['ear'], bins=20, color='blue', alpha=0.7, edgecolor='black')
ax1.axvline(0.25, color='green', linestyle='--', label='Open threshold')
ax1.axvline(0.20, color='red', linestyle='--', label='Closed threshold')
ax1.set_title('Distribution of EAR Values')
ax1.set_xlabel('EAR')
ax1.set_ylabel('Frequency')
ax1.legend()
ax1.grid(True, alpha=0.3)

# باکس‌پلات
ax2.boxplot(df['ear'])
ax2.set_title('Boxplot of EAR')
ax2.set_ylabel('EAR')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ear_analysis.png', dpi=300)
print("\n✅ Chart saved: ear_analysis.png")
plt.show()