import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from database_utils import DAO
from Value_Extraction_Agent.value_extraction_agent import ValueExtractionAgent


def generate_charts(value_distribution):
    df = pd.DataFrame(value_distribution)
    df = df.sort_values(by='Count', ascending=False).reset_index(drop=True)
    palette = sns.color_palette('Blues', 17)[::-1]

    # barchart
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Category', y='Count', data=df, palette="Blues")
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Personal Values')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

    # pie chart
    plt.figure(figsize=(12, 8))
    plt.pie(df['Count'], labels=df['Category'], autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette("Blues", 17))
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    plt.show()


db_helper = DAO()
all_messages = db_helper.get_messages()

generate_charts(all_messages)

random.shuffle(all_messages)
messages = []

pv_agent = ValueExtractionAgent()
pv_agent.optimize_for_recall = True
pv_agent.set_value_extraction_sources(["LLM_7"])

predictions = pv_agent.get_values_for_messages(messages, single_label=False)

# Obtain distribution of values over data set
distribution = {}
for m, values in predictions.items():
    for v in values:
        if v in distribution:
            distribution[v.name].append(m)
        else:
            distribution[v.name] = [m]


print("Distribution:")
print(distribution)

for v, messages in distribution.items():
    print("Value: " + v + ", Number of messages: " + str(len(messages)))


