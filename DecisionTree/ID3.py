from typing import Union
from collections import Counter
from math import log2

class Node:

    def __init__(self, classifier: Union[str, tuple[str, dict[str, "Node"], str]]):
        self.classifier = classifier

    def classify(self, instance: dict[str, str]) -> str:
        if isinstance(self.classifier, str):
            return self.classifier
        else:
            test_attribute, children, default_value = self.classifier
            return children.get(instance.get(test_attribute, default_value), children["__other__"]).classify(instance)

    def stringify(self, indent_level: int):
        if isinstance(self.classifier, str):
            return self.classifier
        else:
            test_attribute, children, default_value = self.classifier
            indentation = '\t' * indent_level
            return ('\n' if indent_level != 0 else '') + indentation + f"\n{indentation}EL".join(f"IF {test_attribute} = {value}{' (default)' if value == default_value else ''} THEN {children[value].stringify(indent_level + 1)}" for value in children)

class DecisionTree:

    def __init__(self, examples: list[dict[str, str]], target: str):
        self.target = target
        self.root = self.ID3(
            examples,
            target,
            {attr for example in examples for attr in example} - {target}
        )

    def show(self):
        print(f"[Target Attribute is '{self.target}']")
        print(self.root.stringify(0))

    def classify(self, instance: dict[str, str]):
        return f"{self.target} = {self.root.classify(instance)}"

    def ID3(self, examples: list[dict[str, str]], target: str, attrs: set[str]):
        target_count = Counter(example[target] for example in examples)
        if len(target_count) == 1:
            return Node(target_count.popitem()[0])
        elif len(attrs) == 0:
            return Node(target_count.most_common(1)[0][0])
        else:
            optimal_classifier, default_value = self.get_optimal_classifier(examples, target, attrs)
            children: dict[str, Node] = {}
            possible_values = {example.get(optimal_classifier, default_value) for example in examples}
            for value in possible_values:
                sub_group = [example for example in examples if example.get(optimal_classifier, default_value) == value]
                children[value] = self.ID3(
                    sub_group,
                    target,
                    attrs - {optimal_classifier}
                )
            children["__other__"] = Node(target_count.most_common(1)[0][0])
            return Node((optimal_classifier, children, default_value))

    def get_optimal_classifier(self, examples: list[dict[str, str]], target: str, attrs: set[str]):
        information_gain = {
            attr: self.compute_information_gain(examples, attr, target) for attr in attrs
        }
        optimal_classifier = max(information_gain, key=lambda attr: information_gain[attr][0])
        return optimal_classifier, information_gain[optimal_classifier][1]

    def compute_information_gain(self, examples: list[dict[str, str]], attr: str, target: str):
        old_entropy = self.compute_entropy(examples, target)
        possible_values = Counter(example[attr] for example in examples if attr in example)
        default_value = possible_values.most_common(1)[0][0]
        new_entropy = 0
        num_example = len(examples)
        for value in possible_values:
            sub_group = [example for example in examples if example.get(attr, default_value) == value]
            new_entropy += (len(sub_group) / num_example) * self.compute_entropy(sub_group, target)
        return old_entropy - new_entropy, default_value

    def compute_entropy(self, examples: list[dict[str, str]], target: str):
        target_value_count = Counter(example[target] for example in examples)
        num_example = len(examples)
        return sum(-(count/num_example)*log2(count/num_example) for count in target_value_count.values())

if __name__ == "__main__":

    # PLayTennis

    EXAMPLES = [
        {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "PlayTennis": "No"},
        {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong", "PlayTennis": "No"},
        {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "High", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "No"},
        {"Outlook": "Overcast", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "Yes"},
        {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "High", "Wind": "Weak", "PlayTennis": "No"},
        {"Outlook": "Sunny", "Temperature": "Cool", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Sunny", "Temperature": "Mild", "Humidity": "Normal", "Wind": "Strong", "PlayTennis": "Yes"},
        {"Outlook": "Overcast", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "PlayTennis": "Yes"},
        {"Outlook": "Overcast", "Temperature": "Hot", "Humidity": "Normal", "Wind": "Weak", "PlayTennis": "Yes"},
        {"Outlook": "Rain", "Temperature": "Mild", "Humidity": "High", "Wind": "Strong", "PlayTennis": "No"},
    ]

    DECISION_TREE = DecisionTree(EXAMPLES, "PlayTennis")
    DECISION_TREE.show()

    NEW_INSTANCE = {"Outlook": "Sunny", "Temperature": "Hot", "Humidity": "High", "Wind": "Strong"}
    print(f"Classify: {NEW_INSTANCE}")
    print(f"Classification: {DECISION_TREE.classify(NEW_INSTANCE)}")

    #######################################################################################################################

    # # EnjoySport

    # EXAMPLES = [
    #     {"Sky": "Sunny", "AirTemp": "Hot", "Humidity": "Normal", "Wind": "Strong", "Water": "Warm", "Forecast": "Same", "EnjoySport": "Yes"},
    #     {"Sky": "Sunny", "AirTemp": "Hot", "Humidity": "High", "Wind": "Strong", "Water": "Warm", "Forecast": "Same", "EnjoySport": "Yes"},
    #     {"Sky": "Rainy", "AirTemp": "Cold", "Humidity": "High", "Wind": "Strong", "Water": "Warm", "Forecast": "Change", "EnjoySport": "No"},
    #     {"Sky": "Sunny", "AirTemp": "Hot", "Humidity": "High", "Wind": "Strong", "Water": "Cool", "Forecast": "Change", "EnjoySport": "Yes"},
    #     {"Sky": "Sunny", "AirTemp": "Hot", "Humidity": "Normal", "Wind": "Weak", "Water": "Warm", "Forecast": "Same", "EnjoySport": "No"},
    # ]

    # DECISION_TREE = DecisionTree(EXAMPLES, "EnjoySport")
    # DECISION_TREE.show()

    #######################################################################################################################

    # # Random

    # EXAMPLES = [
    #     {"a1": "T", "a2": "T", "Classification": "+"},
    #     {"a1": "T", "a2": "T", "Classification": "+"},
    #     {"a1": "T", "a2": "F", "Classification": "-"},
    #     {"a1": "F", "a2": "F", "Classification": "+"},
    #     {"a1": "F", "a2": "T", "Classification": "-"},
    #     {"a1": "F", "a2": "T", "Classification": "-"},
    # ]

    # DECISION_TREE = DecisionTree(EXAMPLES, "Classification")
    # DECISION_TREE.show()
