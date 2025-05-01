from MLP_model import (
    setup_model,
    store_model_parameters
)
from train import optimize_model_parameters  # 从 train.py 导入训练函数
from config import config
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# 数据路径
data_dir = "C:/Users/31521/OneDrive/桌面/files/academic/FDU/25春大三下/神经网络与深度学习/Deep_Learning/pj_1/dataset/preprocessed"

X_train = np.load(f"{data_dir}/X_train.npy")
y_train = np.load(f"{data_dir}/y_train.npy")
X_test = np.load(f"{data_dir}/X_test.npy")
y_test = np.load(f"{data_dir}/y_test.npy")

# 超参数
INPUT_DIM = config.INPUT_DIM
hidden_dim = config.hidden_dim
output_dim = config.output_dim
LEARNING_RATE = config.LEARNING_RATE
decay_rate = config.decay_rate
beta = config.beta
batch_size = config.batch_size
EPOCHS = config.EPOCHS
activation = config.activation
dropout_rate = 0.5

# 初始化模型
model = setup_model(INPUT_DIM, hidden_dim, output_dim)

# 使用 train.py 中的优化函数进行模型训练
model, best_model, history_train_losses, history_train_accuracies, history_val_losses, history_val_accuracies = optimize_model_parameters(
    network=model,
    training_data=X_train,
    training_labels=y_train,
    validation_data=X_test,
    validation_labels=y_test,
    lr_initial=LEARNING_RATE,
    lr_decay=decay_rate,
    regularisation_strength=beta,
    dropout_rate=dropout_rate,
    total_epochs=EPOCHS,
    samples_per_batch=batch_size,
    activation_fn=activation
)

# 保存最佳模型
store_model_parameters(best_model, "best_model_sigmoid")

# 绘制训练曲线
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
f.suptitle("Accuracy and Loss", fontsize = 28)
ax1.plot(history_train_losses, label="Training Loss")
ax1.plot(history_val_losses, label="Validation Loss")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(history_train_accuracies, label="Training Accuracy")
ax2.plot(history_val_accuracies, label="Validation Accuracy")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()

ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig("accuracy_and_loss_sigmoid.png")
plt.show()