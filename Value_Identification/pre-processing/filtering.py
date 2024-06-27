import pandas as pd

# loading data
df = pd.read_excel('')  # insert local file
messages = df['text']

# keeping track of messages that were filtered out and the criterion that they did not fulfill
filtered_out = {}

print("Number of original messages:", len(messages))
original_messages = messages

# FILTERING OF MESSAGES

# 1. long messages
max_message_length = 400

long_messages = messages[messages.apply(len) > max_message_length]
filtered_out["too long"] = long_messages

messages = messages[messages.apply(len) <= max_message_length]
print("Number of messages with less than", max_message_length, "characters:", len(messages))


# 2. short messages
min_message_length = 55

short_messages = messages[messages.apply(len) < min_message_length]
filtered_out["too short"] = short_messages

messages = messages[messages.apply(len) >= min_message_length]
print("Number of messages with more than", min_message_length, "characters:", len(messages))


# statistics of filtered out messages
for reason, message_list in filtered_out.items():
    print(len(message_list), "messages, (", len(message_list)/len(original_messages)*100, "% of original message) were filtered out, because:", reason)

# writing messages that fulfill requirements to csv
df = pd.DataFrame({'message': messages})
df.to_csv('messages_for_annotation.csv', index=False)
