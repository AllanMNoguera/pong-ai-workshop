import matplotlib.pyplot as plt
import seaborn as sns

# Function to visualize the weights of the model
def save_model_image(model, episode, state):
    """
    Visualize and save the weight heatmaps for each layer of the model.
    """
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Convert the weights to a NumPy array
            weights = param.detach().numpy()

            # Create a heatmap using seaborn
            plt.figure(figsize=(8, 6))
            sns.heatmap(weights, cmap="viridis", annot=False, cbar=True)
            plt.title(f'Weight Heatmap for {name} at Episode {episode}')
            plt.xlabel('Neuron Index')
            plt.ylabel('Weight Index')

            # Save the heatmap as an image
            plt.savefig(f'heatmap_{name}_episode_{episode}.png')
            plt.close()

            print(f"Saved heatmap for {name} at episode {episode}")
