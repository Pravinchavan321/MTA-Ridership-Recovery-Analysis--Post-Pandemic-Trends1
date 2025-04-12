import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print(pd.__version__)
print(np.__version__)
print(sns.__version__)

df = pd.read_csv('transit_data.tsv',delimiter=',')
df.head()

print(df.columns)

print(df.dtypes)

# Convert Date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
# Verify the conversion
print(df['Date'].dtype)  # Should output: datetime64[ns]
print(df['Date'].head())

df.set_index('Date', inplace=True)
print(df.head())

print(df.isnull().sum())

df.fillna(0, inplace=True)

df.dropna(subset=['Subways: Total Estimated Ridership'], inplace=True)

ridership_cols = [
    'Subways: Total Estimated Ridership',
    'Buses: Total Estimated Ridership',
    'LIRR: Total Estimated Ridership',
    'Metro-North: Total Estimated Ridership',
    # Add other ridership columns as needed
]
for col in ridership_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

percent_cols = [
    'Subways: % of Comparable Pre-Pandemic Day',
    'Buses: % of Comparable Pre-Pandemic Day',
    # Add other percentage columns
]
for col in percent_cols:
    df[col] = df[col].replace('%', '', regex=True).astype(float) / 100

df.drop_duplicates(inplace=True)

print(df[ridership_cols].describe())

df = df[df[ridership_cols].ge(0).all(axis=1)]  # Keep rows where all ridership >= 0

print(df[ridership_cols].describe())
print(df[percent_cols].describe())

monthly_ridership = df[ridership_cols].resample('M').mean()
print(monthly_ridership.head())

print(df[ridership_cols].corr())

print(df[percent_cols].mean())  # Average recovery rate per mode

plt.figure(figsize=(12, 6))
for col in ridership_cols:
    plt.plot(df.index, df[col], label=col)
plt.title('Transit Ridership Over Time')
plt.xlabel('Date')
plt.ylabel('Total Estimated Ridership')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
monthly_ridership.plot()
plt.title('Monthly Average Transit Ridership')
plt.xlabel('Date')
plt.ylabel('Average Ridership')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for col in percent_cols:
    plt.plot(df.index, df[col], label=col)
plt.title('Ridership as % of Pre-Pandemic Levels')
plt.xlabel('Date')
plt.ylabel('% of Pre-Pandemic Day')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df[ridership_cols])
plt.title('Ridership Distribution by Transit Mode')
plt.xticks(rotation=45)
plt.ylabel('Total Estimated Ridership')
plt.show()

print("Average Recovery Rates:")
print(df[percent_cols].mean())

# Save cleaned DataFrame
df.to_csv('cleaned_transit_data.csv')


df.head()



# Resample data to get monthly averages
monthly_avg = df[ridership_cols].resample('M').mean()

# Create a pivot table with months as rows and transit modes as columns
monthly_avg['Month'] = monthly_avg.index.month
monthly_avg['Year'] = monthly_avg.index.year
pivot_table = monthly_avg.pivot_table(index='Month', columns='Year', values=ridership_cols[0])

# Plot heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap='YlGnBu')
plt.title('Monthly Average Ridership Heatmap')
plt.xlabel('Year')
plt.ylabel('Month')
plt.show()


plt.figure(figsize=(12, 6))
sns.violinplot(data=df[ridership_cols], inner='quartile')
plt.title('Ridership Distribution by Transit Mode')
plt.xticks(rotation=45)
plt.ylabel('Total Estimated Ridership')
plt.show()

sns.pairplot(df[ridership_cols])
plt.suptitle('Pairwise Relationships Between Transit Modes', y=1.02)
plt.show()

# Resample data to get monthly sums
monthly_sum = df[ridership_cols].resample('M').sum()

# Plot stacked area chart
monthly_sum.plot.area(figsize=(12, 6), cmap='tab20')
plt.title('Monthly Total Ridership by Transit Mode')
plt.xlabel('Date')
plt.ylabel('Total Estimated Ridership')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()


# Calculate average ridership per mode
average_ridership = df[ridership_cols].mean()

# Plot pie chart
plt.figure(figsize=(8, 8))
plt.pie(
    average_ridership,
    labels=average_ridership.index,
    autopct='%1.1f%%',
    startangle=140
)
plt.title('Average Ridership Share by Transit Mode')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

# Create regression plot
plt.figure(figsize=(10, 6))
sns.regplot(x='Buses: Total Estimated Ridership', y='Subways: Total Estimated Ridership', data=df, line_kws={'color': 'red'})
plt.title('Regression: Subways vs. Buses Ridership')
plt.xlabel('Buses: Total Estimated Ridership')
plt.ylabel('Subways: Total Estimated Ridership')
plt.grid(True)
plt.show()

