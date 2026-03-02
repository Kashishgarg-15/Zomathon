#!/usr/bin/env python3
"""
Food Delivery Dataset - Comprehensive EDA
21,321 food order records from 6 imaginary restaurants in Delhi NCR
Evaluating suitability for add-on recommendation system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from collections import Counter
from datetime import datetime
import re
import warnings
import os
import json

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.bbox'] = 'tight'

DATA_PATH = "/home/rkp/coding/zomatothon/Food_Delivary_dataSet/order_history_kaggle_data.csv"
VIZ_DIR = "/home/rkp/coding/zomatothon/food_delivery_visualizations"
os.makedirs(VIZ_DIR, exist_ok=True)

# ── Load & Preprocess ──────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} cols")

# Parse datetime
df['order_datetime'] = pd.to_datetime(df['Order Placed At'], format='%I:%M %p, %B %d %Y')
df['order_hour'] = df['order_datetime'].dt.hour
df['order_dow'] = df['order_datetime'].dt.dayofweek  # 0=Mon
df['order_date'] = df['order_datetime'].dt.date
df['order_month'] = df['order_datetime'].dt.to_period('M')
df['order_weekday'] = df['order_datetime'].dt.day_name()

# Parse distance → numeric km
def parse_distance(d):
    if pd.isna(d):
        return np.nan
    d = str(d).strip()
    if d.startswith('<'):
        return 0.5  # <1km → 0.5
    m = re.search(r'(\d+)', d)
    return float(m.group(1)) if m else np.nan

df['distance_km'] = df['Distance'].apply(parse_distance)

# Parse item count and item names
def parse_items(s):
    if pd.isna(s):
        return [], 0
    items = [x.strip() for x in str(s).split(',')]
    total_qty = 0
    names = []
    for item in items:
        m = re.match(r'(\d+)\s*x\s*(.*)', item.strip())
        if m:
            total_qty += int(m.group(1))
            names.append(m.group(2).strip())
        else:
            total_qty += 1
            names.append(item.strip())
    return names, total_qty

parsed = df['Items in order'].apply(parse_items)
df['item_names'] = parsed.apply(lambda x: x[0])
df['item_count'] = parsed.apply(lambda x: x[1])

# Meal period classification
def classify_meal(hour):
    if 6 <= hour <= 10: return 'Breakfast'
    elif 11 <= hour <= 14: return 'Lunch'
    elif 15 <= hour <= 16: return 'Snack'
    elif 17 <= hour <= 21: return 'Dinner'
    else: return 'Late Night'

df['meal_period'] = df['order_hour'].apply(classify_meal)

# Discount percentage
df['discount_total'] = (df['Restaurant discount (Promo)'] +
                        df['Restaurant discount (Flat offs, Freebies & others)'] +
                        df['Gold discount'] + df['Brand pack discount'])
df['discount_pct'] = (df['discount_total'] / df['Bill subtotal'] * 100).clip(0, 100)

# Effective price per item
df['price_per_item'] = df['Total'] / df['item_count'].replace(0, np.nan)

# Parse discount type
def classify_discount(d):
    if pd.isna(d): return 'No Discount'
    d = str(d)
    if 'Buy 1 Get 1' in d or 'BOGO' in d: return 'BOGO'
    elif 'Flat' in d and 'off' in d: return 'Flat Off'
    elif '% off' in d: return 'Percentage Off'
    elif 'Free' in d: return 'Freebie'
    else: return 'Other'

df['discount_type'] = df['Discount construct'].apply(classify_discount)

# Delivered subset
delivered = df[df['Order Status'] == 'Delivered'].copy()

print(f"Preprocessing done. Delivered orders: {len(delivered):,}")
print()

# ══════════════════════════════════════════════════════════════════
# CHART 1: Dataset Overview Dashboard
# ══════════════════════════════════════════════════════════════════
print("Chart 1: Dataset Overview...")
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
fig.suptitle("Food Delivery Dataset — Overview Dashboard", fontsize=20, fontweight='bold', y=1.02)

# 1a: Order status
ax1 = fig.add_subplot(gs[0, 0])
status = df['Order Status'].value_counts()
colors_status = ['#4CAF50', '#F44336', '#FF9800', '#9C27B0', '#2196F3', '#607D8B']
ax1.barh(status.index, status.values, color=colors_status[:len(status)])
for i, v in enumerate(status.values):
    ax1.text(v + 50, i, f"{v:,} ({v/len(df)*100:.1f}%)", va='center', fontsize=9)
ax1.set_title("Order Status Distribution")
ax1.set_xlabel("Count")

# 1b: Restaurant count
ax2 = fig.add_subplot(gs[0, 1])
rest = df['Restaurant name'].value_counts()
bars = ax2.barh(rest.index, rest.values, color=plt.cm.Set2(np.linspace(0, 1, len(rest))))
for i, v in enumerate(rest.values):
    ax2.text(v + 50, i, f"{v:,}", va='center', fontsize=9)
ax2.set_title("Orders by Restaurant")

# 1c: Subzone
ax3 = fig.add_subplot(gs[0, 2])
sz = df['Subzone'].value_counts()
ax3.barh(sz.index, sz.values, color=plt.cm.Pastel1(np.linspace(0, 1, len(sz))))
for i, v in enumerate(sz.values):
    ax3.text(v + 30, i, f"{v:,}", va='center', fontsize=8)
ax3.set_title("Orders by Subzone")

# 1d: Daily order trend
ax4 = fig.add_subplot(gs[1, :2])
daily = df.groupby('order_date').size()
ax4.fill_between(daily.index, daily.values, alpha=0.3, color='#2196F3')
ax4.plot(daily.index, daily.values, color='#2196F3', lw=0.8, alpha=0.7)
# Add 7-day moving average
if len(daily) > 7:
    ma7 = daily.rolling(7).mean()
    ax4.plot(ma7.index, ma7.values, color='#F44336', lw=2, label='7-day MA')
    ax4.legend()
ax4.set_title("Daily Order Volume Over Time")
ax4.set_xlabel("Date")
ax4.set_ylabel("Orders")
ax4.tick_params(axis='x', rotation=30)

# 1e: Key metrics
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
metrics = [
    ("Total Orders", f"{len(df):,}"),
    ("Unique Customers", f"{df['Customer ID'].nunique():,}"),
    ("Restaurants", f"{df['Restaurant name'].nunique()}"),
    ("Subzones", f"{df['Subzone'].nunique()}"),
    ("Avg Bill", f"₹{df['Bill subtotal'].mean():,.0f}"),
    ("Avg Total (after disc)", f"₹{df['Total'].mean():,.0f}"),
    ("Avg Items/Order", f"{df['item_count'].mean():.1f}"),
    ("Delivery Rate", f"{(df['Order Status']=='Delivered').mean()*100:.1f}%"),
    ("Avg Distance", f"{df['distance_km'].mean():.1f} km"),
    ("Date Range", f"{df['order_date'].min()} to\n{df['order_date'].max()}"),
]
for i, (k, v) in enumerate(metrics):
    y = 0.95 - i * 0.095
    ax5.text(0.05, y, k + ":", fontsize=10, fontweight='bold', transform=ax5.transAxes, va='top')
    ax5.text(0.7, y, v, fontsize=10, transform=ax5.transAxes, va='top', color='#1565C0')
ax5.set_title("Key Metrics", fontsize=12, fontweight='bold')

# 1f: Missing data
ax6 = fig.add_subplot(gs[2, :])
null_pct = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
null_pct = null_pct[null_pct > 0]
bars = ax6.bar(range(len(null_pct)), null_pct.values,
               color=['#F44336' if v > 50 else '#FF9800' if v > 20 else '#FFC107' for v in null_pct.values])
ax6.set_xticks(range(len(null_pct)))
ax6.set_xticklabels(null_pct.index, rotation=45, ha='right', fontsize=8)
for i, v in enumerate(null_pct.values):
    ax6.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=7)
ax6.set_title("Missing Data Percentage by Column")
ax6.set_ylabel("% Missing")

plt.savefig(f"{VIZ_DIR}/01_dataset_overview.png", dpi=150)
plt.close()
print("  -> 01_dataset_overview.png")


# ══════════════════════════════════════════════════════════════════
# CHART 2: Restaurant Performance Analysis
# ══════════════════════════════════════════════════════════════════
print("Chart 2: Restaurant Performance...")
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Restaurant Performance Analysis", fontsize=20, fontweight='bold', y=1.02)

restaurants = df['Restaurant name'].unique()
rest_colors = dict(zip(restaurants, plt.cm.Set2(np.linspace(0, 1, len(restaurants)))))

# 2a: Avg order value by restaurant
rest_aov = delivered.groupby('Restaurant name')['Total'].mean().sort_values()
axes[0,0].barh(rest_aov.index, rest_aov.values, color=[rest_colors[r] for r in rest_aov.index])
for i, v in enumerate(rest_aov.values):
    axes[0,0].text(v+5, i, f"₹{v:.0f}", va='center', fontsize=10)
axes[0,0].set_title("Avg Order Value (Total)")

# 2b: Avg rating by restaurant
rest_with_rating = delivered.dropna(subset=['Rating'])
if len(rest_with_rating) > 0:
    rest_rating = rest_with_rating.groupby('Restaurant name')['Rating'].agg(['mean','count'])
    rest_rating = rest_rating.sort_values('mean')
    axes[0,1].barh(rest_rating.index, rest_rating['mean'].values,
                   color=[rest_colors[r] for r in rest_rating.index])
    for i, (m, c) in enumerate(zip(rest_rating['mean'].values, rest_rating['count'].values)):
        axes[0,1].text(m+0.02, i, f"{m:.2f} (n={c})", va='center', fontsize=9)
    axes[0,1].set_xlim(0, 5.5)
    axes[0,1].set_title("Avg Rating by Restaurant")

# 2c: Avg items per order
rest_items = delivered.groupby('Restaurant name')['item_count'].mean().sort_values()
axes[0,2].barh(rest_items.index, rest_items.values, color=[rest_colors[r] for r in rest_items.index])
for i, v in enumerate(rest_items.values):
    axes[0,2].text(v+0.02, i, f"{v:.2f}", va='center', fontsize=10)
axes[0,2].set_title("Avg Items per Order")

# 2d: KPT by restaurant
rest_kpt = delivered.dropna(subset=['KPT duration (minutes)'])
rest_kpt_avg = rest_kpt.groupby('Restaurant name')['KPT duration (minutes)'].mean().sort_values()
axes[1,0].barh(rest_kpt_avg.index, rest_kpt_avg.values,
               color=[rest_colors[r] for r in rest_kpt_avg.index])
for i, v in enumerate(rest_kpt_avg.values):
    axes[1,0].text(v+0.2, i, f"{v:.1f} min", va='center', fontsize=10)
axes[1,0].set_title("Avg Kitchen Prep Time (KPT)")

# 2e: Discount % by restaurant
rest_disc = delivered.groupby('Restaurant name')['discount_pct'].mean().sort_values()
axes[1,1].barh(rest_disc.index, rest_disc.values,
               color=[rest_colors[r] for r in rest_disc.index])
for i, v in enumerate(rest_disc.values):
    axes[1,1].text(v+0.2, i, f"{v:.1f}%", va='center', fontsize=10)
axes[1,1].set_title("Avg Discount % by Restaurant")

# 2f: Order ready marking accuracy
rest_ready = delivered.groupby(['Restaurant name', 'Order Ready Marked']).size().unstack(fill_value=0)
rest_ready_pct = rest_ready.div(rest_ready.sum(axis=1), axis=0) * 100
rest_ready_pct = rest_ready_pct.reindex(columns=['Correctly', 'Incorrectly', 'Missed'], fill_value=0)
rest_ready_pct.plot(kind='barh', stacked=True, ax=axes[1,2],
                    color=['#4CAF50', '#FF9800', '#F44336'])
axes[1,2].set_title("Order Ready Marking Accuracy (%)")
axes[1,2].set_xlabel("%")
axes[1,2].legend(fontsize=8, loc='lower right')

plt.savefig(f"{VIZ_DIR}/02_restaurant_performance.png", dpi=150)
plt.close()
print("  -> 02_restaurant_performance.png")


# ══════════════════════════════════════════════════════════════════
# CHART 3: Temporal Ordering Patterns
# ══════════════════════════════════════════════════════════════════
print("Chart 3: Temporal Patterns...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Temporal Ordering Patterns", fontsize=20, fontweight='bold', y=1.02)

# 3a: Hourly distribution
hourly = df['order_hour'].value_counts().sort_index()
axes[0,0].fill_between(hourly.index, hourly.values, alpha=0.3, color='#2196F3')
axes[0,0].plot(hourly.index, hourly.values, 'o-', color='#2196F3', markersize=6, lw=2)
axes[0,0].set_title("Orders by Hour of Day")
axes[0,0].set_xlabel("Hour (0-23)")
axes[0,0].set_ylabel("Number of Orders")
axes[0,0].set_xticks(range(0, 24))
for label, start, end, c in [('Breakfast', 6, 10, '#FFC107'), ('Lunch', 11, 14, '#FF9800'),
                               ('Dinner', 17, 21, '#F44336'), ('Late Night', 22, 24, '#9C27B0')]:
    axes[0,0].axvspan(start, end, alpha=0.1, color=c, label=label)
axes[0,0].legend(loc='upper left', fontsize=8)

# 3b: Day of week
dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
daily_dow = df['order_dow'].value_counts().sort_index()
colors_dow = ['#2196F3' if i < 5 else '#E91E63' for i in range(7)]
axes[0,1].bar(range(7), daily_dow.values, color=colors_dow)
axes[0,1].set_xticks(range(7))
axes[0,1].set_xticklabels(dow_labels)
axes[0,1].set_title("Orders by Day of Week (Mon=0)")
for i, v in enumerate(daily_dow.values):
    axes[0,1].text(i, v+30, f"{v:,}", ha='center', fontsize=9)

# 3c: Heatmap
hour_dow = df.groupby(['order_dow', 'order_hour']).size().unstack(fill_value=0)
sns.heatmap(hour_dow, ax=axes[1,0], cmap='YlOrRd', cbar_kws={'label': 'Order Count'})
axes[1,0].set_title("Heatmap: Day x Hour")
axes[1,0].set_yticklabels(dow_labels, rotation=0)
axes[1,0].set_xlabel("Hour of Day")

# 3d: Monthly trend
monthly = df.groupby(df['order_datetime'].dt.to_period('M')).size()
axes[1,1].bar(range(len(monthly)), monthly.values, color='#9C27B0', edgecolor='white')
axes[1,1].set_xticks(range(len(monthly)))
axes[1,1].set_xticklabels([str(p) for p in monthly.index], rotation=45, fontsize=8)
axes[1,1].set_title("Monthly Order Volume")
for i, v in enumerate(monthly.values):
    axes[1,1].text(i, v+20, f"{v:,}", ha='center', fontsize=8)

plt.savefig(f"{VIZ_DIR}/03_temporal_patterns.png", dpi=150)
plt.close()
print("  -> 03_temporal_patterns.png")


# ══════════════════════════════════════════════════════════════════
# CHART 4: Meal Period Analysis
# ══════════════════════════════════════════════════════════════════
print("Chart 4: Meal Period...")
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Meal Period Analysis", fontsize=20, fontweight='bold', y=1.02)

meal_order = ['Breakfast', 'Lunch', 'Snack', 'Dinner', 'Late Night']
meal_colors = {'Breakfast': '#FFC107', 'Lunch': '#FF9800', 'Snack': '#8BC34A',
               'Dinner': '#F44336', 'Late Night': '#9C27B0'}

# 4a: Pie chart
meal_counts = df['meal_period'].value_counts().reindex(meal_order).fillna(0)
meal_counts_plot = meal_counts[meal_counts > 0]
axes[0,0].pie(meal_counts_plot.values, labels=meal_counts_plot.index, autopct='%1.1f%%',
              colors=[meal_colors[m] for m in meal_counts_plot.index], startangle=90,
              textprops={'fontsize': 10})
axes[0,0].set_title("Order Distribution by Meal Period")

# 4b: Avg order value by meal
meal_aov = delivered.groupby('meal_period')['Total'].mean().reindex(meal_order).fillna(0)
axes[0,1].bar(range(5), meal_aov.values, color=[meal_colors[m] for m in meal_order])
axes[0,1].set_xticks(range(5))
axes[0,1].set_xticklabels(meal_order, rotation=15)
for i, v in enumerate(meal_aov.values):
    if v > 0:
        axes[0,1].text(i, v+5, f"₹{v:.0f}", ha='center', fontsize=10, fontweight='bold')
axes[0,1].set_title("Avg Order Value by Meal Period")
axes[0,1].set_ylabel("₹ Total")

# 4c: Avg items by meal
meal_items = delivered.groupby('meal_period')['item_count'].mean().reindex(meal_order).fillna(0)
axes[0,2].bar(range(5), meal_items.values, color=[meal_colors[m] for m in meal_order])
axes[0,2].set_xticks(range(5))
axes[0,2].set_xticklabels(meal_order, rotation=15)
for i, v in enumerate(meal_items.values):
    axes[0,2].text(i, v+0.02, f"{v:.2f}", ha='center', fontsize=10)
axes[0,2].set_title("Avg Items per Order by Meal")

# 4d: Discount by meal
meal_disc = delivered.groupby('meal_period')['discount_pct'].mean().reindex(meal_order).fillna(0)
axes[1,0].bar(range(5), meal_disc.values, color=[meal_colors[m] for m in meal_order])
axes[1,0].set_xticks(range(5))
axes[1,0].set_xticklabels(meal_order, rotation=15)
for i, v in enumerate(meal_disc.values):
    axes[1,0].text(i, v+0.2, f"{v:.1f}%", ha='center', fontsize=10)
axes[1,0].set_title("Avg Discount % by Meal Period")

# 4e: Restaurant x Meal heatmap
rest_meal = delivered.groupby(['Restaurant name', 'meal_period']).size().unstack(fill_value=0)
rest_meal = rest_meal.reindex(columns=meal_order, fill_value=0)
rest_meal_pct = rest_meal.div(rest_meal.sum(axis=1), axis=0) * 100
sns.heatmap(rest_meal_pct, ax=axes[1,1], cmap='YlOrRd', annot=True, fmt='.1f',
            linewidths=0.5, cbar_kws={'label': '%'})
axes[1,1].set_title("Restaurant x Meal Period (%)")

# 4f: KPT by meal
meal_kpt = delivered.dropna(subset=['KPT duration (minutes)']).groupby('meal_period')['KPT duration (minutes)'].mean().reindex(meal_order).fillna(0)
axes[1,2].bar(range(5), meal_kpt.values, color=[meal_colors[m] for m in meal_order])
axes[1,2].set_xticks(range(5))
axes[1,2].set_xticklabels(meal_order, rotation=15)
for i, v in enumerate(meal_kpt.values):
    axes[1,2].text(i, v+0.2, f"{v:.1f}m", ha='center', fontsize=10)
axes[1,2].set_title("Avg KPT by Meal Period")

plt.savefig(f"{VIZ_DIR}/04_meal_period_analysis.png", dpi=150)
plt.close()
print("  -> 04_meal_period_analysis.png")


# ══════════════════════════════════════════════════════════════════
# CHART 5: Pricing & Revenue Analysis
# ══════════════════════════════════════════════════════════════════
print("Chart 5: Pricing & Revenue...")
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Pricing & Revenue Analysis", fontsize=20, fontweight='bold', y=1.02)

# 5a: Bill subtotal distribution
axes[0,0].hist(delivered['Bill subtotal'], bins=80, color='#2196F3', edgecolor='white', alpha=0.8)
axes[0,0].axvline(delivered['Bill subtotal'].mean(), color='red', ls='--',
                  label=f"Mean: ₹{delivered['Bill subtotal'].mean():.0f}")
axes[0,0].axvline(delivered['Bill subtotal'].median(), color='green', ls='--',
                  label=f"Median: ₹{delivered['Bill subtotal'].median():.0f}")
axes[0,0].set_xlim(0, 3000)
axes[0,0].set_title("Bill Subtotal Distribution")
axes[0,0].legend()

# 5b: Total after discount
axes[0,1].hist(delivered['Total'], bins=80, color='#4CAF50', edgecolor='white', alpha=0.8)
axes[0,1].axvline(delivered['Total'].mean(), color='red', ls='--',
                  label=f"Mean: ₹{delivered['Total'].mean():.0f}")
axes[0,1].axvline(delivered['Total'].median(), color='green', ls='--',
                  label=f"Median: ₹{delivered['Total'].median():.0f}")
axes[0,1].set_xlim(0, 3000)
axes[0,1].set_title("Total (After Discounts) Distribution")
axes[0,1].legend()

# 5c: Discount distribution
disc_nonzero = delivered[delivered['discount_total'] > 0]['discount_total']
axes[0,2].hist(disc_nonzero, bins=60, color='#FF5722', edgecolor='white', alpha=0.8)
axes[0,2].axvline(disc_nonzero.mean(), color='red', ls='--',
                  label=f"Mean: ₹{disc_nonzero.mean():.0f}")
axes[0,2].set_xlim(0, 500)
axes[0,2].set_title(f"Discount Amount (non-zero, n={len(disc_nonzero):,})")
axes[0,2].legend()

# 5d: Discount type distribution
dt = df['discount_type'].value_counts()
axes[1,0].bar(range(len(dt)), dt.values, color=['#607D8B', '#4CAF50', '#FF9800', '#2196F3', '#E91E63'][:len(dt)])
axes[1,0].set_xticks(range(len(dt)))
axes[1,0].set_xticklabels(dt.index, rotation=20)
for i, v in enumerate(dt.values):
    axes[1,0].text(i, v+100, f"{v:,}\n({v/len(df)*100:.1f}%)", ha='center', fontsize=9)
axes[1,0].set_title("Discount Type Distribution")

# 5e: Packaging charges vs subtotal
axes[1,1].scatter(delivered['Bill subtotal'], delivered['Packaging charges'],
                  alpha=0.1, s=5, color='#9C27B0')
axes[1,1].set_xlim(0, 3000)
axes[1,1].set_ylim(0, 150)
axes[1,1].set_title("Packaging Charges vs Subtotal")
axes[1,1].set_xlabel("Bill Subtotal (₹)")
axes[1,1].set_ylabel("Packaging Charges (₹)")

# 5f: Revenue breakdown stacked
rest_revenue = delivered.groupby('Restaurant name').agg(
    subtotal=('Bill subtotal', 'sum'),
    discount=('discount_total', 'sum'),
    packaging=('Packaging charges', 'sum')
).sort_values('subtotal')
x = range(len(rest_revenue))
axes[1,2].barh(rest_revenue.index, rest_revenue['subtotal'], label='Subtotal', color='#2196F3', alpha=0.8)
axes[1,2].barh(rest_revenue.index, -rest_revenue['discount'], label='Discounts', color='#F44336', alpha=0.8)
axes[1,2].barh(rest_revenue.index, rest_revenue['packaging'], left=rest_revenue['subtotal'],
               label='Packaging', color='#FFC107', alpha=0.8)
axes[1,2].set_title("Revenue Breakdown by Restaurant")
axes[1,2].legend(fontsize=9)

plt.savefig(f"{VIZ_DIR}/05_pricing_revenue.png", dpi=150)
plt.close()
print("  -> 05_pricing_revenue.png")


# ══════════════════════════════════════════════════════════════════
# CHART 6: Delivery Performance
# ══════════════════════════════════════════════════════════════════
print("Chart 6: Delivery Performance...")
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Delivery Performance Analysis", fontsize=20, fontweight='bold', y=1.02)

# 6a: Distance distribution
dist_counts = df['Distance'].value_counts()
dist_order = sorted(dist_counts.index, key=lambda x: parse_distance(x))
dist_vals = [dist_counts[d] for d in dist_order]
axes[0,0].bar(range(len(dist_order)), dist_vals, color='#2196F3', edgecolor='white')
axes[0,0].set_xticks(range(len(dist_order)))
axes[0,0].set_xticklabels(dist_order, rotation=60, fontsize=8)
axes[0,0].set_title("Orders by Distance")

# 6b: KPT distribution
kpt = delivered['KPT duration (minutes)'].dropna()
axes[0,1].hist(kpt, bins=60, color='#FF7043', edgecolor='white', alpha=0.8)
axes[0,1].axvline(kpt.mean(), color='red', ls='--', label=f'Mean: {kpt.mean():.1f} min')
axes[0,1].axvline(kpt.median(), color='green', ls='--', label=f'Median: {kpt.median():.1f} min')
axes[0,1].set_xlim(0, 50)
axes[0,1].set_title("Kitchen Prep Time (KPT) Distribution")
axes[0,1].legend()

# 6c: Rider wait time
rwt = delivered['Rider wait time (minutes)'].dropna()
axes[0,2].hist(rwt, bins=60, color='#26A69A', edgecolor='white', alpha=0.8)
axes[0,2].axvline(rwt.mean(), color='red', ls='--', label=f'Mean: {rwt.mean():.1f} min')
axes[0,2].axvline(rwt.median(), color='green', ls='--', label=f'Median: {rwt.median():.1f} min')
axes[0,2].set_xlim(0, 30)
axes[0,2].set_title("Rider Wait Time Distribution")
axes[0,2].legend()

# 6d: KPT vs Distance
kpt_dist = delivered.dropna(subset=['KPT duration (minutes)'])
kpd = kpt_dist.groupby('distance_km')['KPT duration (minutes)'].mean()
axes[1,0].bar(kpd.index, kpd.values, color='#E91E63', width=0.7)
axes[1,0].set_title("Avg KPT vs Distance")
axes[1,0].set_xlabel("Distance (km)")
axes[1,0].set_ylabel("Avg KPT (min)")

# 6e: KPT by hour
kpt_hour = delivered.dropna(subset=['KPT duration (minutes)']).groupby('order_hour')['KPT duration (minutes)'].mean()
axes[1,1].plot(kpt_hour.index, kpt_hour.values, 'o-', color='#FF5722', lw=2, ms=6)
axes[1,1].fill_between(kpt_hour.index, kpt_hour.values, alpha=0.2, color='#FF5722')
axes[1,1].set_title("Avg KPT by Hour of Day")
axes[1,1].set_xlabel("Hour")
axes[1,1].set_ylabel("KPT (min)")

# 6f: Rider wait by hour
rwt_hour = delivered.dropna(subset=['Rider wait time (minutes)']).groupby('order_hour')['Rider wait time (minutes)'].mean()
axes[1,2].plot(rwt_hour.index, rwt_hour.values, 'o-', color='#26A69A', lw=2, ms=6)
axes[1,2].fill_between(rwt_hour.index, rwt_hour.values, alpha=0.2, color='#26A69A')
axes[1,2].set_title("Avg Rider Wait Time by Hour")
axes[1,2].set_xlabel("Hour")
axes[1,2].set_ylabel("Wait Time (min)")

plt.savefig(f"{VIZ_DIR}/06_delivery_performance.png", dpi=150)
plt.close()
print("  -> 06_delivery_performance.png")


# ══════════════════════════════════════════════════════════════════
# CHART 7: Item Analysis
# ══════════════════════════════════════════════════════════════════
print("Chart 7: Item Analysis...")

# Explode item names for frequency analysis
all_items = []
for names in df['item_names']:
    all_items.extend(names)
item_freq = Counter(all_items)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle("Item & Menu Analysis", fontsize=20, fontweight='bold', y=1.02)

# 7a: Item count distribution
axes[0,0].hist(df['item_count'], bins=np.arange(0.5, 20.5, 1), color='#2196F3', edgecolor='white')
axes[0,0].set_title("Items per Order Distribution")
axes[0,0].axvline(df['item_count'].mean(), color='red', ls='--',
                  label=f"Mean: {df['item_count'].mean():.2f}")
axes[0,0].legend()
single = (df['item_count'] == 1).mean() * 100
axes[0,0].text(0.95, 0.95, f"Single-item: {single:.1f}%", transform=axes[0,0].transAxes,
               ha='right', va='top', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 7b: Top 30 most ordered items
top30 = item_freq.most_common(30)
axes[0,1].barh([x[0][:40] for x in top30][::-1], [x[1] for x in top30][::-1],
               color=plt.cm.viridis(np.linspace(0.2, 0.8, 30)))
axes[0,1].set_title("Top 30 Most Ordered Items")
axes[0,1].tick_params(axis='y', labelsize=8)

# 7c: Items per order by restaurant
rest_item_dist = delivered.groupby('Restaurant name')['item_count'].describe()[['mean', '50%', 'std']]
rest_item_dist = rest_item_dist.sort_values('mean')
x = range(len(rest_item_dist))
axes[1,0].barh(rest_item_dist.index, rest_item_dist['mean'],
               xerr=rest_item_dist['std'], color=plt.cm.Set2(np.linspace(0, 1, len(rest_item_dist))),
               capsize=3)
axes[1,0].set_title("Avg Items/Order by Restaurant (±std)")

# 7d: Price per item distribution
ppi = delivered['price_per_item'].dropna()
ppi = ppi[ppi < 2000]
axes[1,1].hist(ppi, bins=80, color='#4CAF50', edgecolor='white', alpha=0.8)
axes[1,1].axvline(ppi.mean(), color='red', ls='--', label=f'Mean: ₹{ppi.mean():.0f}')
axes[1,1].axvline(ppi.median(), color='green', ls='--', label=f'Median: ₹{ppi.median():.0f}')
axes[1,1].set_xlim(0, 1500)
axes[1,1].set_title("Price per Item Distribution")
axes[1,1].legend()

plt.savefig(f"{VIZ_DIR}/07_item_analysis.png", dpi=150)
plt.close()
print("  -> 07_item_analysis.png")


# ══════════════════════════════════════════════════════════════════
# CHART 8: Customer Behavior
# ══════════════════════════════════════════════════════════════════
print("Chart 8: Customer Behavior...")
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Customer Behavior Analysis", fontsize=20, fontweight='bold', y=1.02)

# Customer stats
cust = df.groupby('Customer ID').agg(
    order_count=('Order ID', 'count'),
    total_spend=('Total', 'sum'),
    avg_order_value=('Total', 'mean'),
    unique_restaurants=('Restaurant name', 'nunique'),
    unique_subzones=('Subzone', 'nunique'),
    avg_items=('item_count', 'mean')
).reset_index()

# 8a: Orders per customer
axes[0,0].hist(cust['order_count'], bins=np.arange(0.5, 30.5, 1), color='#2196F3', edgecolor='white')
axes[0,0].set_title("Orders per Customer")
axes[0,0].axvline(cust['order_count'].mean(), color='red', ls='--',
                  label=f"Mean: {cust['order_count'].mean():.1f}")
axes[0,0].legend()
one_time = (cust['order_count'] == 1).mean() * 100
repeat = (cust['order_count'] > 1).mean() * 100
axes[0,0].text(0.95, 0.95, f"One-time: {one_time:.1f}%\nRepeat: {repeat:.1f}%",
               transform=axes[0,0].transAxes, ha='right', va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 8b: Total spend per customer
axes[0,1].hist(cust['total_spend'], bins=80, color='#4CAF50', edgecolor='white')
axes[0,1].set_xlim(0, 10000)
axes[0,1].set_title("Total Spend per Customer")
axes[0,1].axvline(cust['total_spend'].mean(), color='red', ls='--',
                  label=f"Mean: ₹{cust['total_spend'].mean():.0f}")
axes[0,1].legend()

# 8c: Restaurants per customer
rc = cust['unique_restaurants'].value_counts().sort_index()
axes[0,2].bar(rc.index, rc.values, color='#FF7043')
for i, (k, v) in enumerate(zip(rc.index, rc.values)):
    axes[0,2].text(k, v+50, f"{v/len(cust)*100:.1f}%", ha='center', fontsize=9)
axes[0,2].set_title("Unique Restaurants per Customer")

# 8d: Customer segmentation by frequency
def cust_seg(n):
    if n == 1: return 'One-time'
    elif n <= 3: return 'Occasional (2-3)'
    elif n <= 7: return 'Regular (4-7)'
    else: return 'Loyal (8+)'

cust['segment'] = cust['order_count'].apply(cust_seg)
seg_order = ['One-time', 'Occasional (2-3)', 'Regular (4-7)', 'Loyal (8+)']
sc = cust['segment'].value_counts().reindex(seg_order)
seg_colors = ['#FF5722', '#FF9800', '#4CAF50', '#2196F3']
axes[1,0].bar(range(4), sc.values, color=seg_colors)
axes[1,0].set_xticks(range(4))
axes[1,0].set_xticklabels(seg_order, rotation=15)
for i, v in enumerate(sc.values):
    axes[1,0].text(i, v+50, f"{v:,}\n({v/len(cust)*100:.1f}%)", ha='center', fontsize=9)
axes[1,0].set_title("Customer Frequency Segments")

# 8e: AOV by segment
seg_aov = cust.groupby('segment')['avg_order_value'].mean().reindex(seg_order)
axes[1,1].bar(range(4), seg_aov.values, color=seg_colors)
axes[1,1].set_xticks(range(4))
axes[1,1].set_xticklabels(seg_order, rotation=15)
for i, v in enumerate(seg_aov.values):
    axes[1,1].text(i, v+5, f"₹{v:.0f}", ha='center', fontsize=10, fontweight='bold')
axes[1,1].set_title("Avg Order Value by Segment")

# 8f: Scatter - Frequency vs AOV
s = cust.sample(min(5000, len(cust)), random_state=42)
axes[1,2].scatter(s['order_count'], s['avg_order_value'], alpha=0.15, s=10, color='#9C27B0')
axes[1,2].set_xlim(0, 30)
axes[1,2].set_ylim(0, 3000)
axes[1,2].set_title("Order Frequency vs Avg Order Value")
axes[1,2].set_xlabel("Number of Orders")
axes[1,2].set_ylabel("Avg Order Value (₹)")

plt.savefig(f"{VIZ_DIR}/08_customer_behavior.png", dpi=150)
plt.close()
print("  -> 08_customer_behavior.png")


# ══════════════════════════════════════════════════════════════════
# CHART 9: Rating & Review Analysis
# ══════════════════════════════════════════════════════════════════
print("Chart 9: Ratings & Reviews...")
rated = delivered.dropna(subset=['Rating']).copy()

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle(f"Rating & Review Analysis (n={len(rated):,} rated orders, {len(rated)/len(delivered)*100:.1f}% of delivered)", fontsize=16, fontweight='bold', y=1.02)

# 9a: Rating distribution
rating_counts = rated['Rating'].value_counts().sort_index()
colors_rating = ['#F44336', '#FF5722', '#FF9800', '#8BC34A', '#4CAF50']
axes[0,0].bar(rating_counts.index, rating_counts.values, color=colors_rating, width=0.6)
for i, (k, v) in enumerate(zip(rating_counts.index, rating_counts.values)):
    axes[0,0].text(k, v+10, f"{v:,}\n({v/len(rated)*100:.1f}%)", ha='center', fontsize=9)
axes[0,0].set_title("Rating Distribution")
axes[0,0].set_xlabel("Rating")

# 9b: Rating by restaurant
rest_rating_box = rated.groupby('Restaurant name')['Rating'].apply(list).to_dict()
rests_sorted = rated.groupby('Restaurant name')['Rating'].mean().sort_values().index.tolist()
bp = axes[0,1].boxplot([rest_rating_box.get(r, []) for r in rests_sorted],
                       labels=rests_sorted, vert=True, patch_artist=True)
for patch, r in zip(bp['boxes'], rests_sorted):
    patch.set_facecolor(rest_colors.get(r, '#ccc'))
axes[0,1].set_title("Rating by Restaurant")
axes[0,1].tick_params(axis='x', rotation=25, labelsize=8)

# 9c: Rating vs price per item
rated_ppi = rated.dropna(subset=['price_per_item'])
rated_ppi = rated_ppi[rated_ppi['price_per_item'] < 1500]
val_by_rating = rated_ppi.groupby('Rating')['price_per_item'].mean()
axes[0,2].bar(val_by_rating.index, val_by_rating.values, color='#26A69A', width=0.6)
for k, v in zip(val_by_rating.index, val_by_rating.values):
    axes[0,2].text(k, v+3, f"₹{v:.0f}", ha='center', fontsize=10)
axes[0,2].set_title("Avg Price/Item by Rating")

# 9d: Rating vs KPT
rated_kpt = rated.dropna(subset=['KPT duration (minutes)'])
kpt_by_rating = rated_kpt.groupby('Rating')['KPT duration (minutes)'].mean()
axes[1,0].bar(kpt_by_rating.index, kpt_by_rating.values, color='#FF7043', width=0.6)
for k, v in zip(kpt_by_rating.index, kpt_by_rating.values):
    axes[1,0].text(k, v+0.2, f"{v:.1f}m", ha='center', fontsize=10)
axes[1,0].set_title("Avg KPT by Rating")

# 9e: Rating by meal period
meal_rating = rated.groupby('meal_period')['Rating'].mean().reindex(meal_order).fillna(0)
axes[1,1].bar(range(5), meal_rating.values, color=[meal_colors[m] for m in meal_order])
axes[1,1].set_xticks(range(5))
axes[1,1].set_xticklabels(meal_order, rotation=15)
for i, v in enumerate(meal_rating.values):
    axes[1,1].text(i, v+0.02, f"{v:.2f}", ha='center', fontsize=10)
axes[1,1].set_title("Avg Rating by Meal Period")
axes[1,1].set_ylim(0, 5.5)

# 9f: Rating vs discount
disc_bins = pd.cut(rated['discount_pct'], bins=[0, 5, 15, 25, 40, 100], labels=['0-5%', '5-15%', '15-25%', '25-40%', '40%+'])
db_rating = rated.groupby(disc_bins, observed=True)['Rating'].mean()
axes[1,2].bar(range(len(db_rating)), db_rating.values, color='#9C27B0', width=0.6)
axes[1,2].set_xticks(range(len(db_rating)))
axes[1,2].set_xticklabels(db_rating.index, rotation=15)
for i, v in enumerate(db_rating.values):
    axes[1,2].text(i, v+0.02, f"{v:.2f}", ha='center', fontsize=10)
axes[1,2].set_title("Avg Rating by Discount Level")
axes[1,2].set_ylim(0, 5.5)

plt.savefig(f"{VIZ_DIR}/09_rating_review.png", dpi=150)
plt.close()
print("  -> 09_rating_review.png")


# ══════════════════════════════════════════════════════════════════
# CHART 10: Cancellation & Complaint Analysis
# ══════════════════════════════════════════════════════════════════
print("Chart 10: Cancellations & Complaints...")
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Cancellations, Complaints & Quality Issues", fontsize=20, fontweight='bold', y=1.02)

# 10a: Cancellation reasons
cancel = df[df['Cancellation / Rejection reason'].notna()]
cancel_reasons = cancel['Cancellation / Rejection reason'].value_counts()
axes[0,0].barh(cancel_reasons.index, cancel_reasons.values, color='#F44336')
for i, v in enumerate(cancel_reasons.values):
    axes[0,0].text(v+1, i, f"{v} ({v/len(df)*100:.2f}%)", va='center', fontsize=10)
axes[0,0].set_title(f"Cancellation/Rejection Reasons (n={len(cancel)})")

# 10b: Customer complaints
complaints = df[df['Customer complaint tag'].notna()]
comp_counts = complaints['Customer complaint tag'].value_counts()
axes[0,1].barh(comp_counts.index, comp_counts.values,
               color=['#FF5722', '#FF9800', '#FFC107', '#2196F3', '#9C27B0'][:len(comp_counts)])
for i, v in enumerate(comp_counts.values):
    axes[0,1].text(v+1, i, f"{v} ({v/len(df)*100:.2f}%)", va='center', fontsize=10)
axes[0,1].set_title(f"Customer Complaints (n={len(complaints)})")

# 10c: Complaint rate by restaurant
rest_comp_rate = df.groupby('Restaurant name').apply(
    lambda x: (x['Customer complaint tag'].notna().sum() / len(x)) * 100
).sort_values()
axes[1,0].barh(rest_comp_rate.index, rest_comp_rate.values,
               color=[rest_colors[r] for r in rest_comp_rate.index])
for i, v in enumerate(rest_comp_rate.values):
    axes[1,0].text(v+0.05, i, f"{v:.2f}%", va='center', fontsize=10)
axes[1,0].set_title("Complaint Rate by Restaurant")

# 10d: Order Ready Marking
orm = df['Order Ready Marked'].value_counts()
orm_colors = {'Correctly': '#4CAF50', 'Incorrectly': '#FF9800', 'Missed': '#F44336'}
axes[1,1].pie(orm.values, labels=orm.index, autopct='%1.1f%%',
              colors=[orm_colors.get(k, '#ccc') for k in orm.index],
              startangle=90, textprops={'fontsize': 11})
axes[1,1].set_title("Order Ready Marking Accuracy")

plt.savefig(f"{VIZ_DIR}/10_cancellations_complaints.png", dpi=150)
plt.close()
print("  -> 10_cancellations_complaints.png")


# ══════════════════════════════════════════════════════════════════
# CHART 11: Subzone / Geography Analysis
# ══════════════════════════════════════════════════════════════════
print("Chart 11: Subzone Analysis...")
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Subzone / Geography Analysis", fontsize=20, fontweight='bold', y=1.02)

main_subzones = df['Subzone'].value_counts()
main_sz = main_subzones[main_subzones >= 50].index.tolist()
df_sz = df[df['Subzone'].isin(main_sz)]

sz_colors = dict(zip(main_sz, plt.cm.tab10(np.linspace(0, 1, len(main_sz)))))

# 11a: Orders by subzone
axes[0,0].barh(main_subzones.index, main_subzones.values,
               color=plt.cm.Pastel1(np.linspace(0, 1, len(main_subzones))))
for i, v in enumerate(main_subzones.values):
    axes[0,0].text(v+30, i, f"{v:,}", va='center', fontsize=9)
axes[0,0].set_title("Orders by Subzone")

# 11b: AOV by subzone
sz_aov = df_sz.groupby('Subzone')['Total'].mean().sort_values()
axes[0,1].barh(sz_aov.index, sz_aov.values,
               color=[sz_colors.get(s, '#ccc') for s in sz_aov.index])
for i, v in enumerate(sz_aov.values):
    axes[0,1].text(v+5, i, f"₹{v:.0f}", va='center', fontsize=10)
axes[0,1].set_title("Avg Order Value by Subzone")

# 11c: Distance by subzone
sz_dist = df_sz.groupby('Subzone')['distance_km'].mean().sort_values()
axes[0,2].barh(sz_dist.index, sz_dist.values,
               color=[sz_colors.get(s, '#ccc') for s in sz_dist.index])
for i, v in enumerate(sz_dist.values):
    axes[0,2].text(v+0.05, i, f"{v:.1f} km", va='center', fontsize=10)
axes[0,2].set_title("Avg Distance by Subzone")

# 11d: Restaurant x Subzone heatmap
rs_heat = df.groupby(['Restaurant name', 'Subzone']).size().unstack(fill_value=0)
sns.heatmap(rs_heat, ax=axes[1,0], cmap='Blues', annot=True, fmt='d',
            linewidths=0.5, cbar_kws={'label': 'Orders'}, annot_kws={'fontsize': 8})
axes[1,0].set_title("Restaurant x Subzone Matrix")
axes[1,0].tick_params(axis='x', rotation=30, labelsize=8)

# 11e: KPT by subzone
sz_kpt = df_sz.dropna(subset=['KPT duration (minutes)']).groupby('Subzone')['KPT duration (minutes)'].mean().sort_values()
axes[1,1].barh(sz_kpt.index, sz_kpt.values,
               color=[sz_colors.get(s, '#ccc') for s in sz_kpt.index])
for i, v in enumerate(sz_kpt.values):
    axes[1,1].text(v+0.1, i, f"{v:.1f}m", va='center', fontsize=10)
axes[1,1].set_title("Avg KPT by Subzone")

# 11f: Hourly pattern by subzone
for sz in main_sz[:5]:
    szd = df_sz[df_sz['Subzone'] == sz]
    hourly_sz = szd['order_hour'].value_counts().sort_index()
    hourly_sz = hourly_sz / hourly_sz.sum()
    axes[1,2].plot(hourly_sz.index, hourly_sz.values, 'o-', label=sz[:15], ms=3, lw=1.5)
axes[1,2].set_title("Hourly Pattern by Subzone (top 5)")
axes[1,2].set_xlabel("Hour")
axes[1,2].legend(fontsize=8)

plt.savefig(f"{VIZ_DIR}/11_subzone_analysis.png", dpi=150)
plt.close()
print("  -> 11_subzone_analysis.png")


# ══════════════════════════════════════════════════════════════════
# CHART 12: Discount Effectiveness
# ══════════════════════════════════════════════════════════════════
print("Chart 12: Discount Effectiveness...")
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Discount Strategy & Effectiveness", fontsize=20, fontweight='bold', y=1.02)

# 12a: Discount type vs AOV
dt_aov = delivered.groupby('discount_type')['Total'].mean().sort_values()
axes[0,0].barh(dt_aov.index, dt_aov.values, color=plt.cm.Set3(np.linspace(0, 1, len(dt_aov))))
for i, v in enumerate(dt_aov.values):
    axes[0,0].text(v+5, i, f"₹{v:.0f}", va='center', fontsize=10)
axes[0,0].set_title("Avg Order Value by Discount Type")

# 12b: Discount type vs items
dt_items = delivered.groupby('discount_type')['item_count'].mean().sort_values()
axes[0,1].barh(dt_items.index, dt_items.values, color=plt.cm.Set3(np.linspace(0, 1, len(dt_items))))
for i, v in enumerate(dt_items.values):
    axes[0,1].text(v+0.02, i, f"{v:.2f}", va='center', fontsize=10)
axes[0,1].set_title("Avg Items by Discount Type")

# 12c: Discount % distribution
axes[0,2].hist(delivered[delivered['discount_pct'] > 0]['discount_pct'], bins=50,
               color='#FF5722', edgecolor='white', alpha=0.8)
axes[0,2].set_title("Discount % Distribution (non-zero)")
axes[0,2].set_xlabel("Discount %")

# 12d: Discount % vs Total scatter
sample_d = delivered.sample(min(5000, len(delivered)), random_state=42)
axes[1,0].scatter(sample_d['discount_pct'], sample_d['Total'], alpha=0.1, s=5, color='#9C27B0')
axes[1,0].set_xlim(0, 60)
axes[1,0].set_ylim(0, 3000)
axes[1,0].set_title("Discount % vs Total Order Value")
axes[1,0].set_xlabel("Discount %")
axes[1,0].set_ylabel("Total (₹)")

# 12e: Discount by restaurant x type
rest_dt = delivered.groupby(['Restaurant name', 'discount_type']).size().unstack(fill_value=0)
rest_dt_pct = rest_dt.div(rest_dt.sum(axis=1), axis=0) * 100
rest_dt_pct.plot(kind='bar', stacked=True, ax=axes[1,1], colormap='Set3')
axes[1,1].set_title("Discount Type Mix by Restaurant")
axes[1,1].legend(fontsize=7, loc='upper right')
axes[1,1].tick_params(axis='x', rotation=25, labelsize=8)

# 12f: BOGO analysis
bogo = delivered[delivered['discount_type'] == 'BOGO']
non_bogo = delivered[delivered['discount_type'] != 'BOGO']
categories = ['Avg Total', 'Avg Subtotal', 'Avg Items', 'Avg Discount%']
bogo_vals = [bogo['Total'].mean(), bogo['Bill subtotal'].mean(), bogo['item_count'].mean(), bogo['discount_pct'].mean()]
non_bogo_vals = [non_bogo['Total'].mean(), non_bogo['Bill subtotal'].mean(), non_bogo['item_count'].mean(), non_bogo['discount_pct'].mean()]
x = np.arange(4)
axes[1,2].bar(x-0.2, bogo_vals, 0.35, label=f'BOGO (n={len(bogo):,})', color='#E91E63')
axes[1,2].bar(x+0.2, non_bogo_vals, 0.35, label=f'Non-BOGO (n={len(non_bogo):,})', color='#2196F3')
axes[1,2].set_xticks(x)
axes[1,2].set_xticklabels(categories, fontsize=9)
axes[1,2].legend()
axes[1,2].set_title("BOGO vs Non-BOGO Comparison")

plt.savefig(f"{VIZ_DIR}/12_discount_effectiveness.png", dpi=150)
plt.close()
print("  -> 12_discount_effectiveness.png")


# ══════════════════════════════════════════════════════════════════
# CHART 13: Customer Loyalty & Retention
# ══════════════════════════════════════════════════════════════════
print("Chart 13: Customer Loyalty...")
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Customer Loyalty & Retention Analysis", fontsize=20, fontweight='bold', y=1.02)

# Customer timeline
cust_timeline = df.sort_values('order_datetime').groupby('Customer ID').agg(
    first_order=('order_datetime', 'min'),
    last_order=('order_datetime', 'max'),
    order_count=('Order ID', 'count'),
    total_spend=('Total', 'sum'),
    avg_total=('Total', 'mean')
).reset_index()
cust_timeline['tenure_days'] = (cust_timeline['last_order'] - cust_timeline['first_order']).dt.days
cust_timeline['first_month'] = cust_timeline['first_order'].dt.to_period('M')

# 13a: Tenure distribution (repeat customers only)
repeat_cust = cust_timeline[cust_timeline['order_count'] > 1]
axes[0,0].hist(repeat_cust['tenure_days'], bins=50, color='#2196F3', edgecolor='white')
axes[0,0].set_title(f"Customer Tenure (Repeat, n={len(repeat_cust):,})")
axes[0,0].set_xlabel("Days Between First & Last Order")
axes[0,0].axvline(repeat_cust['tenure_days'].mean(), color='red', ls='--',
                  label=f"Mean: {repeat_cust['tenure_days'].mean():.0f} days")
axes[0,0].legend()

# 13b: Customer lifetime value
clv = cust_timeline['total_spend'].sort_values(ascending=False)
cum_clv = clv.cumsum() / clv.sum() * 100
axes[0,1].plot(range(1, len(cum_clv)+1), cum_clv.values, color='#E91E63', lw=2)
axes[0,1].axhline(80, color='gray', ls='--', alpha=0.5)
axes[0,1].axhline(50, color='gray', ls='--', alpha=0.5)
i80 = np.searchsorted(cum_clv.values, 80)
axes[0,1].annotate(f'80% revenue from top {i80:,}\n({i80/len(clv)*100:.1f}% of customers)',
                   (i80, 80), fontsize=9, color='red')
axes[0,1].set_title("Customer Lifetime Value (CLV) Concentration")
axes[0,1].set_xlabel("Customers (ranked by spend)")
axes[0,1].set_ylabel("Cumulative % of Revenue")

# 13c: New customers per month
new_per_month = cust_timeline.groupby('first_month').size()
axes[0,2].bar(range(len(new_per_month)), new_per_month.values, color='#4CAF50', edgecolor='white')
axes[0,2].set_xticks(range(len(new_per_month)))
axes[0,2].set_xticklabels([str(p) for p in new_per_month.index], rotation=45, fontsize=8)
axes[0,2].set_title("New Customers per Month")

# 13d: Repeat rate by restaurant
repeat_by_rest = df.groupby(['Customer ID', 'Restaurant name']).size().reset_index(name='visits')
rest_repeat = repeat_by_rest.groupby('Restaurant name').apply(
    lambda x: (x['visits'] > 1).mean() * 100
).sort_values()
axes[1,0].barh(rest_repeat.index, rest_repeat.values,
               color=[rest_colors[r] for r in rest_repeat.index])
for i, v in enumerate(rest_repeat.values):
    axes[1,0].text(v+0.3, i, f"{v:.1f}%", va='center', fontsize=10)
axes[1,0].set_title("Customer Repeat Rate by Restaurant")

# 13e: Average days between orders for repeat customers
cust_orders = df.sort_values('order_datetime').groupby('Customer ID')['order_datetime'].apply(list)
gaps = []
for orders_list in cust_orders:
    if len(orders_list) > 1:
        for i in range(1, len(orders_list)):
            gap = (orders_list[i] - orders_list[i-1]).days
            if gap >= 0:
                gaps.append(gap)

if gaps:
    axes[1,1].hist(gaps, bins=np.arange(0, 60, 1), color='#9C27B0', edgecolor='white', alpha=0.8)
    axes[1,1].axvline(np.mean(gaps), color='red', ls='--', label=f'Mean: {np.mean(gaps):.1f} days')
    axes[1,1].axvline(np.median(gaps), color='green', ls='--', label=f'Median: {np.median(gaps):.1f} days')
    axes[1,1].set_xlim(0, 60)
    axes[1,1].set_title("Days Between Consecutive Orders")
    axes[1,1].legend()

# 13f: Order frequency over tenure
repeat_cust_sorted = repeat_cust.sort_values('tenure_days')
axes[1,2].scatter(repeat_cust['tenure_days'], repeat_cust['order_count'],
                  alpha=0.15, s=10, color='#FF7043')
axes[1,2].set_title("Orders vs Customer Tenure")
axes[1,2].set_xlabel("Tenure (days)")
axes[1,2].set_ylabel("Order Count")

plt.savefig(f"{VIZ_DIR}/13_customer_loyalty.png", dpi=150)
plt.close()
print("  -> 13_customer_loyalty.png")


# ══════════════════════════════════════════════════════════════════
# CHART 14: Food Category / Menu Intelligence
# ══════════════════════════════════════════════════════════════════
print("Chart 14: Menu Intelligence...")

# Classify items into food categories
def classify_food(name):
    name_l = str(name).lower()
    if any(w in name_l for w in ['pizza', 'margherita', 'margarita', 'cheesy']):
        return 'Pizza'
    elif any(w in name_l for w in ['burger', 'bun']):
        return 'Burger'
    elif any(w in name_l for w in ['chicken', 'tandoori', 'tangdi', 'tikka', 'kebab', 'wings']):
        return 'Chicken'
    elif any(w in name_l for w in ['paneer', 'dal', 'naan', 'roti', 'paratha', 'kulcha', 'biryani', 'rice', 'raita', 'curry']):
        return 'Indian Main'
    elif any(w in name_l for w in ['fries', 'wedges', 'nachos', 'momos', 'starter', 'soup']):
        return 'Sides/Starters'
    elif any(w in name_l for w in ['shake', 'mojito', 'drink', 'cold coffee', 'lassi', 'lemonade', 'juice', 'tea', 'chai', 'water', 'pepsi', 'coke', 'soda', 'beverage']):
        return 'Beverages'
    elif any(w in name_l for w in ['brownie', 'cake', 'ice cream', 'dessert', 'gulab jamun', 'sweet', 'pastry', 'cookie', 'chocolate']):
        return 'Desserts'
    elif any(w in name_l for w in ['wrap', 'roll', 'sandwich', 'sub ']):
        return 'Wraps/Rolls'
    elif any(w in name_l for w in ['salad', 'coleslaw']):
        return 'Salads'
    elif any(w in name_l for w in ['sauce', 'dip', 'mayo', 'ketchup', 'chutney']):
        return 'Condiments'
    elif any(w in name_l for w in ['combo', 'meal', 'box']):
        return 'Combos'
    else:
        return 'Other'

# Explode items with categories
item_records = []
for _, row in df.iterrows():
    for item_name in row['item_names']:
        item_records.append({
            'item_name': item_name,
            'restaurant': row['Restaurant name'],
            'category': classify_food(item_name),
            'meal_period': row['meal_period'],
            'order_hour': row['order_hour'],
            'total': row['Total']
        })

items_df = pd.DataFrame(item_records)

fig, axes = plt.subplots(2, 3, figsize=(22, 16))
fig.suptitle("Food Category & Menu Intelligence", fontsize=20, fontweight='bold', y=1.02)

# 14a: Category distribution
cat_counts = items_df['category'].value_counts()
axes[0,0].barh(cat_counts.index, cat_counts.values,
               color=plt.cm.Set3(np.linspace(0, 1, len(cat_counts))))
for i, v in enumerate(cat_counts.values):
    axes[0,0].text(v+50, i, f"{v:,} ({v/len(items_df)*100:.1f}%)", va='center', fontsize=9)
axes[0,0].set_title("Food Category Distribution")

# 14b: Category by restaurant
cat_rest = items_df.groupby(['restaurant', 'category']).size().unstack(fill_value=0)
cat_rest_pct = cat_rest.div(cat_rest.sum(axis=1), axis=0) * 100
# Keep top 8 categories
top_cats = cat_counts.head(8).index.tolist()
cat_rest_pct[top_cats].plot(kind='barh', stacked=True, ax=axes[0,1], colormap='Set3')
axes[0,1].set_title("Category Mix by Restaurant")
axes[0,1].legend(fontsize=7, bbox_to_anchor=(1, 1))

# 14c: Category by meal period
cat_meal = items_df.groupby(['meal_period', 'category']).size().unstack(fill_value=0)
cat_meal = cat_meal.reindex(meal_order, fill_value=0)
cat_meal_pct = cat_meal.div(cat_meal.sum(axis=1), axis=0) * 100
cat_meal_pct[top_cats].plot(kind='bar', stacked=True, ax=axes[0,2], colormap='Set3')
axes[0,2].set_title("Category Mix by Meal Period")
axes[0,2].legend(fontsize=7, bbox_to_anchor=(1, 1))
axes[0,2].tick_params(axis='x', rotation=15)

# 14d: Top items by category (top 5 categories)
for idx, cat in enumerate(cat_counts.head(3).index):
    ax = axes[1, idx]
    cat_items = items_df[items_df['category'] == cat]['item_name'].value_counts().head(15)
    ax.barh([n[:35] for n in cat_items.index][::-1], cat_items.values[::-1],
            color=plt.cm.viridis(np.linspace(0.2, 0.8, len(cat_items))))
    ax.set_title(f"Top Items: {cat}")
    ax.tick_params(axis='y', labelsize=7)

plt.savefig(f"{VIZ_DIR}/14_menu_intelligence.png", dpi=150)
plt.close()
print("  -> 14_menu_intelligence.png")


# ══════════════════════════════════════════════════════════════════
# CHART 15: Co-ordering Patterns (Add-on Potential)
# ══════════════════════════════════════════════════════════════════
print("Chart 15: Co-ordering Patterns...")
from itertools import combinations

# Build co-occurrence from item categories per order
order_cats = []
for _, row in df.iterrows():
    cats = set(classify_food(n) for n in row['item_names'])
    if len(cats) > 1:
        order_cats.append(cats)

cat_co = Counter()
for cats in order_cats:
    for c1, c2 in combinations(sorted(cats), 2):
        cat_co[(c1, c2)] += 1

# Build item co-occurrence (top 50 items)
top_item_names = Counter(all_items).most_common(50)
top_item_set = set(n for n, _ in top_item_names)

item_co = Counter()
for _, row in df.iterrows():
    order_items = [n for n in row['item_names'] if n in top_item_set]
    if len(order_items) > 1:
        for i1, i2 in combinations(sorted(set(order_items)), 2):
            item_co[(i1, i2)] += 1

fig, axes = plt.subplots(2, 2, figsize=(20, 18))
fig.suptitle("Co-ordering Patterns (Add-on Recommendation Potential)", fontsize=18, fontweight='bold', y=1.02)

# 15a: Category co-occurrence heatmap
all_cats = sorted(set(c for pair in cat_co for c in pair))
co_mat = pd.DataFrame(0, index=all_cats, columns=all_cats)
for (c1, c2), count in cat_co.items():
    co_mat.loc[c1, c2] = count
    co_mat.loc[c2, c1] = count
sns.heatmap(co_mat, ax=axes[0,0], cmap='YlOrRd', annot=True, fmt='d',
            linewidths=0.5, annot_kws={'fontsize': 7})
axes[0,0].set_title("Food Category Co-occurrence")
axes[0,0].tick_params(axis='x', rotation=45, labelsize=8)
axes[0,0].tick_params(axis='y', labelsize=8)

# 15b: Top category pairs
top_cat_pairs = sorted(cat_co.items(), key=lambda x: x[1], reverse=True)[:20]
axes[0,1].barh([f"{p[0]} + {p[1]}" for p, _ in top_cat_pairs][::-1],
               [c for _, c in top_cat_pairs][::-1],
               color=plt.cm.Spectral(np.linspace(0, 1, 20)))
axes[0,1].set_title("Top 20 Category Co-occurrences")
axes[0,1].tick_params(axis='y', labelsize=8)

# 15c: Top item pairs
top_item_pairs = sorted(item_co.items(), key=lambda x: x[1], reverse=True)[:25]
axes[1,0].barh([f"{p[0][:25]} + {p[1][:25]}" for p, _ in top_item_pairs][::-1],
               [c for _, c in top_item_pairs][::-1],
               color=plt.cm.plasma(np.linspace(0.2, 0.8, 25)))
axes[1,0].set_title("Top 25 Item Co-occurrences")
axes[1,0].tick_params(axis='y', labelsize=6)

# 15d: Multi-category orders analysis
multi_cat_counts = []
for _, row in df.iterrows():
    cats = set(classify_food(n) for n in row['item_names'])
    multi_cat_counts.append(len(cats))
df['category_count'] = multi_cat_counts

cc = pd.Series(multi_cat_counts).value_counts().sort_index()
axes[1,1].bar(cc.index, cc.values, color='#2196F3', edgecolor='white')
for k, v in zip(cc.index, cc.values):
    axes[1,1].text(k, v+100, f"{v:,}\n({v/len(df)*100:.1f}%)", ha='center', fontsize=9)
axes[1,1].set_title("Number of Food Categories per Order")
axes[1,1].set_xlabel("Category Count")

plt.savefig(f"{VIZ_DIR}/15_co_ordering_patterns.png", dpi=150)
plt.close()
print("  -> 15_co_ordering_patterns.png")


# ══════════════════════════════════════════════════════════════════
# CHART 16: Word Cloud & Item Popularity
# ══════════════════════════════════════════════════════════════════
print("Chart 16: Word Clouds...")
from wordcloud import WordCloud

fig, axes = plt.subplots(2, 2, figsize=(20, 14))
fig.suptitle("Item Name Word Clouds & Text Analysis", fontsize=18, fontweight='bold', y=1.02)

# 16a: All items
wc1 = WordCloud(width=800, height=400, background_color='white', colormap='viridis', max_words=150)
all_text = ' '.join(all_items)
wc1.generate(all_text)
axes[0,0].imshow(wc1)
axes[0,0].axis('off')
axes[0,0].set_title("All Ordered Items")

# 16b: Top restaurant (Aura Pizzas)
aura_items = []
for _, row in df[df['Restaurant name'] == 'Aura Pizzas'].iterrows():
    aura_items.extend(row['item_names'])
wc2 = WordCloud(width=800, height=400, background_color='white', colormap='magma', max_words=150)
wc2.generate(' '.join(aura_items))
axes[0,1].imshow(wc2)
axes[0,1].axis('off')
axes[0,1].set_title("Aura Pizzas Items")

# 16c: Swaad
swaad_items = []
for _, row in df[df['Restaurant name'] == 'Swaad'].iterrows():
    swaad_items.extend(row['item_names'])
wc3 = WordCloud(width=800, height=400, background_color='white', colormap='plasma', max_words=150)
wc3.generate(' '.join(swaad_items))
axes[1,0].imshow(wc3)
axes[1,0].axis('off')
axes[1,0].set_title("Swaad Items")

# 16d: High-rated orders
high_rated = delivered[delivered['Rating'] >= 4]
hr_items = []
for _, row in high_rated.iterrows():
    hr_items.extend(row['item_names'])
if hr_items:
    wc4 = WordCloud(width=800, height=400, background_color='white', colormap='cool', max_words=150)
    wc4.generate(' '.join(hr_items))
    axes[1,1].imshow(wc4)
    axes[1,1].axis('off')
    axes[1,1].set_title("Items in High-Rated (4-5★) Orders")

plt.savefig(f"{VIZ_DIR}/16_wordclouds.png", dpi=150)
plt.close()
print("  -> 16_wordclouds.png")


# ══════════════════════════════════════════════════════════════════
# CHART 17: Delivery Ops & Efficiency
# ══════════════════════════════════════════════════════════════════
print("Chart 17: Operational Efficiency...")
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Delivery Operations & Efficiency", fontsize=20, fontweight='bold', y=1.02)

# 17a: KPT vs Rider Wait scatter
kpt_rw = delivered.dropna(subset=['KPT duration (minutes)', 'Rider wait time (minutes)'])
axes[0,0].scatter(kpt_rw['KPT duration (minutes)'], kpt_rw['Rider wait time (minutes)'],
                  alpha=0.08, s=5, color='#2196F3')
axes[0,0].set_xlim(0, 50)
axes[0,0].set_ylim(0, 30)
axes[0,0].set_title("KPT vs Rider Wait Time")
axes[0,0].set_xlabel("KPT (min)")
axes[0,0].set_ylabel("Rider Wait (min)")

# 17b: High KPT analysis
kpt_data = delivered.dropna(subset=['KPT duration (minutes)'])
kpt_buckets = pd.cut(kpt_data['KPT duration (minutes)'], bins=[0, 10, 15, 20, 25, 30, 100],
                     labels=['0-10', '10-15', '15-20', '20-25', '25-30', '30+'])
kb = kpt_buckets.value_counts().reindex(['0-10', '10-15', '15-20', '20-25', '25-30', '30+'])
axes[0,1].bar(range(len(kb)), kb.values,
              color=['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#FF5722', '#F44336'])
axes[0,1].set_xticks(range(len(kb)))
axes[0,1].set_xticklabels(kb.index)
for i, v in enumerate(kb.values):
    axes[0,1].text(i, v+50, f"{v:,}\n({v/len(kpt_data)*100:.1f}%)", ha='center', fontsize=9)
axes[0,1].set_title("KPT Distribution Buckets")

# 17c: Order Ready Marking impact on Wait Time
orm_wait = delivered.dropna(subset=['Rider wait time (minutes)'])
orm_groups = orm_wait.groupby('Order Ready Marked')['Rider wait time (minutes)'].agg(['mean', 'median', 'count'])
bars = axes[0,2].bar(range(len(orm_groups)), orm_groups['mean'],
                     color=['#4CAF50', '#FF9800', '#F44336'][:len(orm_groups)])
axes[0,2].set_xticks(range(len(orm_groups)))
axes[0,2].set_xticklabels(orm_groups.index)
for i, (m, c) in enumerate(zip(orm_groups['mean'], orm_groups['count'])):
    axes[0,2].text(i, m+0.1, f"{m:.1f}m\n(n={c:,})", ha='center', fontsize=9)
axes[0,2].set_title("Avg Rider Wait by Order Ready Marking")

# 17d: Distance vs KPT
dist_kpt = delivered.dropna(subset=['KPT duration (minutes)']).groupby('distance_km').agg(
    mean_kpt=('KPT duration (minutes)', 'mean'),
    mean_rwt=('Rider wait time (minutes)', 'mean')
)
axes[1,0].plot(dist_kpt.index, dist_kpt['mean_kpt'], 'o-', color='#FF5722', label='KPT', lw=2)
axes[1,0].plot(dist_kpt.index, dist_kpt['mean_rwt'], 's-', color='#26A69A', label='Rider Wait', lw=2)
axes[1,0].set_title("KPT & Rider Wait vs Distance")
axes[1,0].set_xlabel("Distance (km)")
axes[1,0].set_ylabel("Minutes")
axes[1,0].legend()

# 17e: Items vs KPT
item_kpt = delivered.dropna(subset=['KPT duration (minutes)']).groupby('item_count')['KPT duration (minutes)'].mean()
item_kpt = item_kpt[item_kpt.index <= 10]
axes[1,1].bar(item_kpt.index, item_kpt.values, color='#9C27B0', edgecolor='white')
axes[1,1].set_title("Avg KPT vs Item Count")
axes[1,1].set_xlabel("Items in Order")
axes[1,1].set_ylabel("KPT (min)")

# 17f: Efficiency score (KPT / item count)
eff = delivered.dropna(subset=['KPT duration (minutes)'])
eff_by_rest = eff.groupby('Restaurant name').apply(
    lambda x: x['KPT duration (minutes)'].mean() / x['item_count'].mean()
).sort_values()
axes[1,2].barh(eff_by_rest.index, eff_by_rest.values,
               color=[rest_colors[r] for r in eff_by_rest.index])
for i, v in enumerate(eff_by_rest.values):
    axes[1,2].text(v+0.1, i, f"{v:.1f} min/item", va='center', fontsize=10)
axes[1,2].set_title("Kitchen Efficiency (KPT per Item)")

plt.savefig(f"{VIZ_DIR}/17_operational_efficiency.png", dpi=150)
plt.close()
print("  -> 17_operational_efficiency.png")


# ══════════════════════════════════════════════════════════════════
# CHART 18: Peak Hour & Demand Forecasting Insights
# ══════════════════════════════════════════════════════════════════
print("Chart 18: Demand Patterns...")
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.suptitle("Peak Hour & Demand Patterns", fontsize=20, fontweight='bold', y=1.02)

# 18a: Orders per hour by restaurant
for r in restaurants:
    rd = df[df['Restaurant name'] == r]
    hourly_r = rd['order_hour'].value_counts().sort_index()
    if len(hourly_r) > 3:
        axes[0,0].plot(hourly_r.index, hourly_r.values, 'o-', label=r, ms=3, lw=1.5)
axes[0,0].set_title("Hourly Orders by Restaurant")
axes[0,0].set_xlabel("Hour")
axes[0,0].legend(fontsize=7)

# 18b: Weekend vs Weekday
is_weekend = df['order_dow'].isin([5, 6])
we = df[is_weekend]
wd = df[~is_weekend]
we_hourly = we['order_hour'].value_counts().sort_index() / we['order_date'].nunique()
wd_hourly = wd['order_hour'].value_counts().sort_index() / wd['order_date'].nunique()
axes[0,1].plot(we_hourly.index, we_hourly.values, 'o-', label='Weekend', color='#E91E63', lw=2)
axes[0,1].plot(wd_hourly.index, wd_hourly.values, 'o-', label='Weekday', color='#2196F3', lw=2)
axes[0,1].set_title("Avg Hourly Orders: Weekend vs Weekday")
axes[0,1].legend()

# 18c: AOV by hour
aov_hour = delivered.groupby('order_hour')['Total'].mean()
axes[0,2].plot(aov_hour.index, aov_hour.values, 'o-', color='#4CAF50', lw=2, ms=6)
axes[0,2].fill_between(aov_hour.index, aov_hour.values, alpha=0.2, color='#4CAF50')
axes[0,2].set_title("Avg Order Value by Hour")
axes[0,2].set_xlabel("Hour")
axes[0,2].set_ylabel("₹ Total")

# 18d: Items by hour
items_hour = delivered.groupby('order_hour')['item_count'].mean()
axes[1,0].plot(items_hour.index, items_hour.values, 'o-', color='#FF5722', lw=2, ms=6)
axes[1,0].fill_between(items_hour.index, items_hour.values, alpha=0.2, color='#FF5722')
axes[1,0].set_title("Avg Items per Order by Hour")
axes[1,0].set_xlabel("Hour")

# 18e: Revenue by hour (orders * avg_total)
rev_hour = delivered.groupby('order_hour')['Total'].sum()
axes[1,1].bar(rev_hour.index, rev_hour.values / 1000, color='#9C27B0', edgecolor='white')
axes[1,1].set_title("Total Revenue by Hour (₹K)")
axes[1,1].set_xlabel("Hour")
axes[1,1].set_ylabel("Revenue (₹K)")

# 18f: Peak hour identification
peak_data = df.groupby(['order_date', 'order_hour']).size().reset_index(name='orders')
peak_95 = peak_data.groupby('order_hour')['orders'].quantile(0.95)
peak_mean = peak_data.groupby('order_hour')['orders'].mean()
axes[1,2].fill_between(peak_95.index, peak_95.values, alpha=0.3, color='#F44336', label='95th pctile')
axes[1,2].plot(peak_95.index, peak_95.values, '--', color='#F44336', lw=1)
axes[1,2].plot(peak_mean.index, peak_mean.values, 'o-', color='#2196F3', lw=2, label='Mean')
axes[1,2].set_title("Peak vs Average Hourly Demand")
axes[1,2].set_xlabel("Hour")
axes[1,2].set_ylabel("Orders per Hour")
axes[1,2].legend()

plt.savefig(f"{VIZ_DIR}/18_demand_patterns.png", dpi=150)
plt.close()
print("  -> 18_demand_patterns.png")


# ══════════════════════════════════════════════════════════════════
# CHART 19: Dataset Suitability Radar
# ══════════════════════════════════════════════════════════════════
print("Chart 19: Suitability Assessment...")

categories_radar = ['Cart\nComposition', 'Sequential\nBuilding', 'Temporal\nPatterns',
                    'User\nHistory', 'Co-purchase', 'Cold\nStart',
                    'Price\nData', 'Restaurant\nContext', 'Geography', 'Meal\nTypes']
N = len(categories_radar)

food_delivery = [7, 2, 9, 7, 6, 6, 9, 8, 5, 8]
instacart_raw = [9, 9, 8, 9, 9, 7, 2, 2, 1, 3]
combined = [9, 7, 9, 9, 8, 7, 8, 8, 5, 8]
ideal = [10] * 10

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
for lst in [food_delivery, instacart_raw, combined, ideal]:
    lst.append(lst[0])
angles.append(angles[0])

fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
ax.fill(angles, ideal, alpha=0.05, color='gray')
ax.plot(angles, ideal, 'o--', lw=1, color='gray', label='Ideal', ms=3)

ax.fill(angles, instacart_raw, alpha=0.15, color='#FF5722')
ax.plot(angles, instacart_raw, 'o-', lw=2, color='#FF5722', label='Instacart Raw', ms=5)

ax.fill(angles, food_delivery, alpha=0.15, color='#2196F3')
ax.plot(angles, food_delivery, 'o-', lw=2, color='#2196F3', label='Food Delivery Dataset', ms=5)

ax.fill(angles, combined, alpha=0.15, color='#4CAF50')
ax.plot(angles, combined, 'o-', lw=2, color='#4CAF50', label='Combined (Both)', ms=5)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=10)
ax.set_ylim(0, 10)
ax.set_yticks([2, 4, 6, 8, 10])
ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)
ax.set_title("Dataset Suitability for Add-on Recommendation System", fontsize=14,
             fontweight='bold', y=1.08)

plt.savefig(f"{VIZ_DIR}/19_suitability_radar.png", dpi=150)
plt.close()
print("  -> 19_suitability_radar.png")


# ══════════════════════════════════════════════════════════════════
# CHART 20: Cross-Dataset Comparison with Instacart
# ══════════════════════════════════════════════════════════════════
print("Chart 20: Cross-Dataset Comparison...")

fig, ax = plt.subplots(figsize=(16, 10))

features = [
    'Total Records', 'Unique Users', 'Restaurants/Stores', 'Geographic Zones',
    'Price Data', 'Discount Data', 'Item-level Detail', 'Sequential Cart',
    'Time of Day', 'Day of Week', 'Delivery Metrics', 'Ratings/Reviews',
    'Customer Complaints', 'Prep Time (KPT)', 'Rider Wait Time',
    'Cuisine Classification', 'Co-purchase Mining', 'User Segmentation'
]

instacart_scores = [10, 10, 1, 1, 1, 1, 10, 10, 8, 8, 1, 1, 1, 1, 1, 2, 10, 9]
fd_scores =       [3,  6,  8, 7, 9, 9,  7,  2, 9, 9, 9, 5, 7, 9, 9, 7,  6, 7]

y = np.arange(len(features))
bars1 = ax.barh(y - 0.2, instacart_scores, 0.35, label='Instacart', color='#FF5722', alpha=0.8)
bars2 = ax.barh(y + 0.2, fd_scores, 0.35, label='Food Delivery', color='#2196F3', alpha=0.8)

ax.set_yticks(y)
ax.set_yticklabels(features, fontsize=9)
ax.set_xlabel("Score (1-10)")
ax.set_xlim(0, 12)
ax.legend(fontsize=12, loc='lower right')
ax.set_title("Feature Comparison: Instacart vs Food Delivery Dataset",
             fontsize=16, fontweight='bold')

# Highlight complementary features
for i, (is_score, fd_score) in enumerate(zip(instacart_scores, fd_scores)):
    if abs(is_score - fd_score) >= 5:
        ax.axhspan(i-0.45, i+0.45, alpha=0.08, color='green')

ax.text(11, len(features)-1, "Green = Highly\nComplementary", fontsize=8,
        color='green', ha='center', va='center',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.savefig(f"{VIZ_DIR}/20_cross_dataset_comparison.png", dpi=150)
plt.close()
print("  -> 20_cross_dataset_comparison.png")


# ══════════════════════════════════════════════════════════════════
# Save metrics JSON
# ══════════════════════════════════════════════════════════════════
print("\nSaving metrics...")
metrics = {
    "dataset_summary": {
        "total_orders": len(df),
        "delivered_orders": len(delivered),
        "unique_customers": int(df['Customer ID'].nunique()),
        "unique_restaurants": int(df['Restaurant name'].nunique()),
        "unique_subzones": int(df['Subzone'].nunique()),
        "date_range": f"{df['order_date'].min()} to {df['order_date'].max()}",
        "delivery_rate_pct": round((df['Order Status'] == 'Delivered').mean() * 100, 2),
    },
    "pricing": {
        "avg_subtotal": round(df['Bill subtotal'].mean(), 2),
        "median_subtotal": round(df['Bill subtotal'].median(), 2),
        "avg_total": round(df['Total'].mean(), 2),
        "avg_discount_pct": round(df['discount_pct'].mean(), 2),
        "avg_packaging": round(df['Packaging charges'].mean(), 2),
    },
    "items": {
        "avg_items_per_order": round(df['item_count'].mean(), 2),
        "single_item_pct": round((df['item_count'] == 1).mean() * 100, 2),
        "unique_items": len(item_freq),
        "top_5_items": [name for name, _ in item_freq.most_common(5)],
    },
    "delivery": {
        "avg_distance_km": round(df['distance_km'].mean(), 2),
        "avg_kpt_min": round(kpt.mean(), 2),
        "median_kpt_min": round(kpt.median(), 2),
        "avg_rider_wait_min": round(rwt.mean(), 2),
        "correct_ready_marking_pct": round((df['Order Ready Marked'] == 'Correctly').mean() * 100, 2),
    },
    "customers": {
        "avg_orders_per_customer": round(cust['order_count'].mean(), 2),
        "repeat_customer_pct": round(repeat * 1, 2),  # already in %
        "one_time_customer_pct": round(one_time * 1, 2),
        "avg_total_spend": round(cust['total_spend'].mean(), 2),
    },
    "ratings": {
        "rated_pct": round(len(rated) / len(delivered) * 100, 2),
        "avg_rating": round(rated['Rating'].mean(), 2) if len(rated) > 0 else None,
        "5star_pct": round((rated['Rating'] == 5).mean() * 100, 2) if len(rated) > 0 else None,
    },
    "suitability_scores": {
        "cart_composition": 7, "sequential_building": 2, "temporal_patterns": 9,
        "user_history": 7, "co_purchase": 6, "cold_start": 6,
        "price_data": 9, "restaurant_context": 8, "geography": 5, "meal_types": 8,
        "overall": 6.7,
        "note": "Complementary to Instacart - strong in pricing/restaurant/delivery, weak in sequential cart data"
    }
}

with open(f"{VIZ_DIR}/analysis_metrics.json", 'w') as f:
    json.dump(metrics, f, indent=2, default=str)
print(f"  -> analysis_metrics.json")


# ══════════════════════════════════════════════════════════════════
# Final Summary
# ══════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"ANALYSIS COMPLETE")
print(f"{'='*70}")
print(f"Generated 20 visualizations in: {VIZ_DIR}/")
print(f"Files:")
for f in sorted(os.listdir(VIZ_DIR)):
    size = os.path.getsize(os.path.join(VIZ_DIR, f))
    print(f"  {f:50s} {size/1024:>8.1f} KB")
print(f"{'='*70}")
