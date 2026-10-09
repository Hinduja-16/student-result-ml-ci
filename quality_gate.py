
import json
import sys

with open("metrics.json", "r") as file:
    metrics = json.load(file)

accuracy = metrics["accuracy"]
minimum_accuracy = 0.60

print("Model accuracy:", accuracy)
print("Minimum required accuracy:", minimum_accuracy)

if accuracy < minimum_accuracy:
    print("Quality gate failed!")
    sys.exit(1)

print("Quality gate passed!")
