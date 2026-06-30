import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# پیدا کردن آخرین فایل
files = glob.glob('fatigue_data_*/session_*.csv')
if files:
    latest = sorted(files)[-1]
    print(f"📂 Reading: {latest}")
    
    df = pd.read_csv(latest)
    
    # آمار
    print("\n📊 Statistics:")
    print(f"Total frames: {len(df)}")
    print(f"Duration: {len(df)/30:.1f} seconds")
    print(f"Mean EAR: {df['ear'].mean():.3f}")
    print(f"Min EAR: {df['ear'].min():.3f}")
    
    print("\n📈 States:")
    print(df['state'].value_counts())
    
    # نمودار
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(df['ear'], color='blue', linewidth=2)
    ax1.set_title('EAR Over Time')
    ax1.set_ylabel('EAR')
    ax1.grid(True)
    
    ax2.plot(df['drowsy_score'], color='red', linewidth=2)
    ax2.set_title('Drowsy Score')
    ax2.set_ylabel('Score %')
    ax2.set_xlabel('Frame')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
else:
    print("❌ No data found!")