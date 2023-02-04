from typing import Iterable
from itertools import product

class Taxonomy:

    def __init__(self, name: str, sub_groups: list["Taxonomy"]):
        self.name = name
        self.sub_groups = sub_groups
        self.parent_groups: list[Taxonomy] = []
        for node in self.sub_groups:
            node.parent_groups.append(self)

    def __gt__(self, other: "Taxonomy"):
        parents = other.parent_groups
        while parents != []:
            if self in parents:
                return True
            parents = list(set(
                new_parent for parent in parents for new_parent in parent.parent_groups
            ))
        else:
            return False

    def __ge__(self, other: "Taxonomy"):
        return self is other or self > other

    def __repr__(self):
        return self.name

    def show(self):
        for sub_group in self.sub_groups:
            print(f"{self.name}->{sub_group.name}")
            sub_group.show()

class Hypothesis:

    def __init__(self, constraints: list[Taxonomy]):
        self.constraints = constraints

    def cover(self, instance: list[Taxonomy]):
        assert len(self.constraints) == len(instance), f"Unequal length of constraints and attributes: {self}, {instance}"
        return all(constraint >= attribute for constraint, attribute in zip(self.constraints, instance))

    def __ge__(self, other: "Hypothesis"):
        return all(a >= b for a, b in zip(self.constraints, other.constraints))

    def __gt__(self, other: "Hypothesis"):
        return self >= other and not other >= self

    def __hash__(self):
        return hash(tuple(self.constraints))

    def __eq__(self, other: object):
        return isinstance(other, Hypothesis) and self.constraints == other.constraints

    def minimal_specializations(self, negative_example: list[Taxonomy], S: "BoundarySet"):
        for i, (constraint, attribute) in enumerate(zip(self.constraints, negative_example)):
            more_specific_constraints = constraint.sub_groups
            end_loop = False
            while more_specific_constraints != []:
                for more_specific_constraint in more_specific_constraints:
                    if not more_specific_constraint >= attribute:
                        end_loop = True
                        new_constraints = self.constraints.copy()
                        new_constraints[i] = more_specific_constraint
                        new_hypothesis = Hypothesis(new_constraints)
                        if not new_hypothesis.cover(negative_example) and any(new_hypothesis > s for s in S.members):
                            yield new_hypothesis
                if end_loop:
                    break
                more_specific_constraints = list(set(
                    new_sub_group for sub_group in more_specific_constraints for new_sub_group in sub_group.sub_groups
                ))

    def minimal_generalizations(self, positive_example: list[Taxonomy], G: "BoundarySet"):
        possible_new_constraints: list[list[Taxonomy]] = [[] for _ in range(len(self.constraints))]
        for i, (constraint, attribute) in enumerate(zip(self.constraints, positive_example)):
            if constraint >= attribute:
                possible_new_constraints[i].append(constraint)
                continue
            more_general_constraints = constraint.parent_groups
            end_loop = False
            while more_general_constraints != []:
                for more_general_constraint in more_general_constraints:
                    if more_general_constraint >= attribute:
                        end_loop = True
                        possible_new_constraints[i].append(more_general_constraint)
                if end_loop:
                    break
                more_general_constraints: list[Taxonomy] = list(set(
                    new_parent for parent in more_general_constraints for new_parent in parent.parent_groups
                ))
        for minimal_generalization in product(*possible_new_constraints):
            new_hypothesis = Hypothesis(list(minimal_generalization))
            if new_hypothesis.cover(positive_example) and any(g > new_hypothesis for g in G.members):
                yield new_hypothesis

    def __repr__(self):
        return f"<{', '.join(constraint.name for constraint in self.constraints)}>"

class BoundarySet:

    def __init__(self, init_members: list[Hypothesis]):
        self.members = init_members

    def remove_inconsistency(self, example: list[Taxonomy], classification: bool):
        if classification is True:
            self.members = [hypothesis for hypothesis in self.members if hypothesis.cover(example)]
        else:
            self.members = [hypothesis for hypothesis in self.members if not hypothesis.cover(example)]

    def generalize(self, positive_example: list[Taxonomy], G: "BoundarySet"):
        new_S_members: list[Hypothesis] = []
        for s in self.members:
            if not s.cover(positive_example):
                for new_s in s.minimal_generalizations(positive_example, G):
                    new_S_members.append(new_s)
            else:
                new_S_members.append(s)
        self.members = [s for s in new_S_members if not any(s > another for another in new_S_members)]

    def specialize(self, negative_example: list[Taxonomy], S: "BoundarySet"):
        new_G_members: list[Hypothesis] = []
        for g in self.members:
            if g.cover(negative_example):
                for new_g in g.minimal_specializations(negative_example, S):
                    new_G_members.append(new_g)
            else:
                new_G_members.append(g)
        self.members = [g for g in new_G_members if not any(another > g for another in new_G_members)]

    def __repr__(self):
        return f"{{{', '.join(str(hypothesis) for hypothesis in self.members)}}}"

class VersionSpace:

    def __init__(self, init_S: BoundarySet, init_G: BoundarySet):
        self.S = init_S
        self.G = init_G

    def learn(self, examples: Iterable[tuple[list[Taxonomy], bool]]):
        S, G = self.S, self.G
        print("Initial Version Space")
        print(f"S0: {S}")
        print(f"G0: {G}\n")
        for i, (example, classification) in enumerate(examples, start=1):
            if classification is True:
                G.remove_inconsistency(example, classification)
                S.generalize(example, G)
            else:
                S.remove_inconsistency(example, classification)
                G.specialize(example, S)
            print(f"{i}: {'+' if classification is True else '-'}{example}")
            print(f"S{i}: {S}")
            print(f"G{i}: {G}\n")
        self.generate_intermediate_hypotheses()
        print("Final Version Space")
        self.show()

    def generate_intermediate_hypotheses(self):
        S, G = self.S, self.G
        self.hypotheses = [G.members]
        base_layer = G.members
        middle_layer: set[Hypothesis] = set()
        generated: set[Hypothesis] = set()
        while True:
            for b in base_layer:
                for i, constraint in enumerate(b.constraints):
                    more_specific_constraints = constraint.sub_groups
                    for more_specific_constraint in more_specific_constraints:
                        new_constraints = b.constraints.copy()
                        new_constraints[i] = more_specific_constraint
                        new_hypothesis = Hypothesis(new_constraints)
                        if new_hypothesis not in generated and any(new_hypothesis > s for s in S.members):
                            middle_layer.add(new_hypothesis)
            if len(middle_layer) == 0:
                break
            else:
                self.hypotheses.append(list(middle_layer))
                generated.update(middle_layer)
                base_layer, middle_layer = middle_layer, set()
        self.hypotheses.append(S.members)

    def show(self):
        if not hasattr(self, "hypotheses"):
            self.generate_intermediate_hypotheses()
        print("Most General")
        for layer in self.hypotheses:
            print(", ".join(str(hypothesis) for hypothesis in layer))
        print("Most Specific\n")

    def classify(self, new_instance: list[Taxonomy]):
        """
        return
            classification: bool
            confidence: float
        """
        if not hasattr(self, "hypotheses"):
            self.generate_intermediate_hypotheses()
        positive, all = 0, 0
        for layer in self.hypotheses:
            for hypothesis in layer:
                if hypothesis.cover(new_instance):
                    positive += 1
                all += 1
        if positive * 2 > all:
            return True, positive / all
        else:
            return False, (all - positive) / all

if __name__ == "__main__":

    # EnjoySport

    # Sky
    NoSky = Taxonomy("_", [])
    Sunny = Taxonomy("Sunny", [NoSky])
    Cloudy = Taxonomy("Cloudy", [NoSky])
    Rainy = Taxonomy("Rainy", [NoSky])
    AnySky = Taxonomy("?", [Sunny, Cloudy, Rainy])

    # AirTemp
    NoAirTemp = Taxonomy("_", [])
    Hot = Taxonomy("Hot", [NoAirTemp])
    Cold = Taxonomy("Cold", [NoAirTemp])
    AnyAirTemp = Taxonomy("?", [Hot, Cold])

    # Humidity
    NoHumidity = Taxonomy("_", [])
    Normal = Taxonomy("Normal", [NoHumidity])
    High = Taxonomy("High", [NoHumidity])
    AnyHumidity = Taxonomy("?", [Normal, High])

    # Wind
    NoWind = Taxonomy("_", [])
    Strong = Taxonomy("Strong", [NoWind])
    Weak = Taxonomy("Weak", [NoWind])
    AnyWind = Taxonomy("?", [Strong, Weak])

    # Water
    NoWater = Taxonomy("_", [])
    Warm = Taxonomy("Warm", [NoWater])
    Cool = Taxonomy("Cool", [NoWater])
    AnyWater = Taxonomy("?", [Warm, Cool])

    # Forecast
    NoForecast = Taxonomy("_", [])
    Same = Taxonomy("Same", [NoForecast])
    Change = Taxonomy("Change", [NoForecast])
    AnyForecast = Taxonomy("?", [Same, Change])

    EXAMPLES = [
        ([Sunny, Hot, Normal, Strong, Warm, Same], True),
        ([Sunny, Hot, High, Strong, Warm, Same], True),
        ([Rainy, Cold, High, Strong, Warm, Change], False),
        ([Sunny, Hot, High, Strong, Cool, Change], True),
    ]

    S = BoundarySet([Hypothesis([NoSky, NoAirTemp, NoHumidity, NoWind, NoWater, NoForecast])])
    G = BoundarySet([Hypothesis([AnySky, AnyAirTemp, AnyHumidity, AnyWind, AnyWater, AnyForecast])])
    VS = VersionSpace(S, G)
    VS.learn(EXAMPLES)

    NEW_INSTANCES = [
        [Sunny, Hot, Normal, Strong, Cool, Change],
        [Rainy, Cold, Normal, Weak, Warm, Same],
        [Sunny, Hot, Normal, Weak, Warm, Same],
        [Sunny, Cold, Normal, Strong, Warm, Same],
    ]
    for new_instance in NEW_INSTANCES:
        classification, confidence = VS.classify(new_instance)
        print(f"Classify: {new_instance}")
        print(f"Classification: {'Positive' if classification is True else 'Negative'}")
        print(f"Confidence: {confidence}\n")

    ##########################################################################################################

    # # LiveTogether

    # # Sex
    # NoSex = Taxonomy("_", [])
    # Male = Taxonomy("Male", [NoSex])
    # Female = Taxonomy("Female", [NoSex])
    # AnySex = Taxonomy("?", [Male, Female])

    # # Hair
    # NoHair = Taxonomy("_", [])
    # Black = Taxonomy("Black", [NoHair])
    # Brown = Taxonomy("Brown", [NoHair])
    # Blonde = Taxonomy("Blonde", [NoHair])
    # AnyHair = Taxonomy("?", [Black, Brown, Blonde])

    # # Height
    # NoHeight = Taxonomy("_", [])
    # Tall = Taxonomy("Tall", [NoHeight])
    # Medium = Taxonomy("Medium", [NoHeight])
    # Short = Taxonomy("Short", [NoHeight])
    # AnyHeight = Taxonomy("?", [Tall, Medium, Short])

    # # Nationality
    # NoNationality = Taxonomy("_", [])
    # American = Taxonomy("American", [NoNationality])
    # French = Taxonomy("French", [NoNationality])
    # German = Taxonomy("German", [NoNationality])
    # Irish = Taxonomy("Irish", [NoNationality])
    # Indian = Taxonomy("Indian", [NoNationality])
    # Japanese = Taxonomy("Japanese", [NoNationality])
    # Portuguese = Taxonomy("Portuguese", [NoNationality])
    # AnyNationality = Taxonomy("?", [American, French, German, Irish, Indian, Japanese, Portuguese])

    # EXAMPLES = [
    #     ([Male, Brown, Tall, American, Female, Black, Short, American], True),
    #     ([Male, Brown, Short, French, Female, Black, Short, American], True),
    #     ([Female, Brown, Tall, German, Female, Black, Short, Indian], False),
    #     ([Male, Brown, Tall, Irish, Female, Brown, Short, Irish], True),
    # ]

    # S = BoundarySet([Hypothesis([NoSex, NoHair, NoHeight, NoNationality, NoSex, NoHair, NoHeight, NoNationality])])
    # G = BoundarySet([Hypothesis([AnySex, AnyHair, AnyHeight, AnyNationality, AnySex, AnyHair, AnyHeight, AnyNationality])])
    # VS = VersionSpace(S, G)
    # VS.learn(EXAMPLES)

    # NEW_INSTANCE = [Male, Black, Short, Portuguese, Female, Blonde, Tall, Indian]
    # classification, confidence = VS.classify(NEW_INSTANCE)
    # print(f"Classify: {NEW_INSTANCE}")
    # print(f"Classification: {'Positive' if classification is True else 'Negative'}")
    # print(f"Confidence: {confidence}")
