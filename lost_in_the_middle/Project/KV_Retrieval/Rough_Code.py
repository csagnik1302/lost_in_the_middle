import matplotlib.pyplot as plt

with open("/home/irlab/sagnik/lost_in_the_middle/Project/Plots/generated_output_kv_140.txt","r") as f:
    out=f.readlines()


out=[int(i) for i in out]

print(out)

## Plot
positions = [1, 35, 70, 105, 140]

accuracies = [100 * x / 500 for x in out]

plt.figure(figsize=(8, 5))

plt.plot(
    positions,
    accuracies,
    marker="o"
)


plt.xticks(positions)

plt.xlabel("Gold KV Position")
plt.ylabel("Accuracy (%)")
plt.title("Llama 3.1 8B KV Retrieval Accuracy vs Position")

plt.ylim(0, 100)

plt.grid(True)

plt.savefig("/home/irlab/sagnik/lost_in_the_middle/Project/Plots/kv_retrieval_accuracy_140.png", dpi=300, bbox_inches="tight")

plt.show()
