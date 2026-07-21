import matplotlib.pyplot as plt

with open("/home/irlab/sagnik/lost_in_the_middle/Project/Plots/generated_output_qa_30.txt","r") as f:
    out=f.readlines()


out=[int(i[:-3]) for i in out]


## Plot

positions = [1, 5, 10, 15, 20, 25, 30]

accuracies = [100 * x / 2655 for x in out]

plt.figure(figsize=(8, 5))

plt.plot(
    positions,
    accuracies,
    marker="o"
)

plt.xticks(positions)

plt.xlabel("Gold QA Position")
plt.ylabel("Accuracy (%)")
plt.title("Llama 3.1 8B QA Retrieval Accuracy vs Position")

plt.ylim(0, 100)

plt.grid(True)

plt.savefig("/home/irlab/sagnik/lost_in_the_middle/Project/Plots/qa_retrieval_accuracy_30_docs.png", dpi=300, bbox_inches="tight")

plt.show()