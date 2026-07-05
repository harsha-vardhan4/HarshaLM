from training.dataset import ConversationDataset

tokens = list(range(20))

dataset = ConversationDataset(
    tokens,
    context_length=5
)

print("Dataset Size:", len(dataset))

input_tokens, target_tokens = dataset[0]

print()

print("Input")

print(input_tokens)

print()

print("Target")

print(target_tokens)