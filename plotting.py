import  matplotlib.pyplot as plt  
import json

with open('output.txt') as f:
    tmp = f.read()


x = json.loads(tmp)
train_loss = []
val_loss = []
for i in range(len(x)):
    epoch = (x[str(i)])
    train_loss.append(epoch['Train Loss'])
    val_loss.append(epoch['Val Loss'])


plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss', marker='o')
plt.plot(range(1, len(val_loss) + 1), val_loss, label='Validation Loss', marker='s')

# Add labels and title
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True, alpha=0.3)

# Optional: Set y-axis to start from 0 for better visualization
plt.ylim(bottom=0)

# Display the plot
plt.tight_layout()
plt.savefig('result.png')
plt.show()
