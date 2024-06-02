from sentence_transformers import SentenceTransformer, losses, InputExample
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from random import random
device = 'cuda'


def random_drop(text: str):
    return ' '.join([t if random() > 0.5 else '' for t in text.split()])

num_epochs = 1
batch_size = 32
warmup_steps = 100  # 10% of train data for warm-up


train_dataset = load_dataset("wikipedia", "20220301.simple", split="train[:1000]")
train_data = [InputExample(texts=[s, random_drop(s)]) for s in tqdm(train_dataset['text'])]
print(train_data[0])
print(f"train dataset size: {len(train_dataset)}")
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, drop_last=True)


test_dataset = load_dataset("wikipedia", "20220301.simple", split="train[20000:20500]")
test_data = [InputExample(texts=[random_drop(s), random_drop(s)]) for s in test_dataset['text']]
print(f"test dataset size: {len(test_dataset)}")
val_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, drop_last=True)

model_name = "all-MiniLM-L12-v2"
model = SentenceTransformer(model_name).to(device)

train_loss = losses.MultipleNegativesRankingLoss(model)


model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=num_epochs,
    warmup_steps=warmup_steps,
    show_progress_bar=True
)
model.save(f'{model_name}-1')