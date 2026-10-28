
from App.llm import generate_preference

question = "Which wins in standard Rock-Paper-Scissors?"

first = generate_preference(
    question, "Scissors", "Paper"
)

second = generate_preference(
    question, "Paper", "Scissors"
)
third=generate_preference(
    question, "Rock", "Paper"
)
fourth=generate_preference(
    question, "Paper", "Rock"
)
fifth=generate_preference(
    question, "Rock", "Scissors"
)
sixth=generate_preference(
    question, "Scissors", "Rock"
)
winner_first = "Scissors" if first == "A" else "Paper"
winner_second = "Paper" if second == "A" else "Scissors"
winner_third = "Rock" if third == "A" else "Paper"
winner_fourth = "Paper" if fourth == "A" else "Rock"
winner_fifth = "Rock" if fifth == "A" else "Scissors"
winner_sixth = "Scissors" if sixth == "A" else "Rock"

print("Original-order winner:", winner_first)
print("Reversed-order winner:", winner_second)
print("Third-order winner:", winner_third)
print("Fourth-order winner:", winner_fourth)
print("Fifth-order winner:", winner_fifth)
print("Sixth-order winner:", winner_sixth)
print("Order stable:", winner_first == winner_second)
print("Order stable:", winner_third == winner_fourth)
print("Order stable:", winner_fifth == winner_sixth)