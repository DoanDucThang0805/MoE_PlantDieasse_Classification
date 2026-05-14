from sklearn.model_selection import train_test_split

# Dữ liệu giả
image_paths = [f"img_{i}.jpg" for i in range(10)]
labels = [0,0,0,0,0,1,1,1,1,1]

train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths,
    labels,
    test_size=0.3,
    stratify=labels,
    random_state=42,
    shuffle=True
)

print("TRAIN:")
print(train_paths)

print("\nTEST:")
print(test_paths)