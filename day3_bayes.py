"""
DAY 3: Probability - Two Problems in One
1. Bayes Theorem - Disease Test Problem
2. Red King in Deck of Cards
"""


# ============================================================
# PROBLEM 1: Bayes Theorem - Disease Test
# ============================================================

print("\n" + "=" * 60)
print("PROBLEM 1: BAYES THEOREM - DISEASE TEST")
print("=" * 60)

# Given data
prevalence = 0.02  # 2% population has disease
accuracy = 0.95    # Test is 95% accurate

# Calculations
p_disease = prevalence
p_healthy = 1 - prevalence

p_pos_given_disease = accuracy
p_pos_given_healthy = 1 - accuracy  # False positive rate = 5%

# Bayes Theorem Formula
p_disease_given_pos = (p_pos_given_disease * p_disease) / (
    p_pos_given_disease * p_disease + p_pos_given_healthy * p_healthy
)

print(f"\n Problem: 2% population has virus. Test is 95% accurate.")
print(f" You test positive. What's the probability you actually have it?\n")

print(f"   P(Disease) = {p_disease * 100}%")
print(f"   P(Positive | Disease) = {p_pos_given_disease * 100}%")
print(f"   P(Positive | Healthy) = {p_pos_given_healthy * 100}%")
print(f"\n P(Disease | Positive) = {p_disease_given_pos * 100:.1f}%")

# Explanation
print(f"\n Explanation: Despite 95% accuracy, only {p_disease_given_pos * 100:.1f}% chance!")
print(f"   Because disease is rare (2%), most positives are false positives.")

# ============================================================
# PROBLEM 2: Red King in Deck of Cards
# ============================================================

print("\n" + "=" * 60)
print("PROBLEM 2: RED KING IN A DECK OF CARDS")
print("=" * 60)

# Given data
total_cards = 52
red_kings = 2  # King of Hearts + King of Diamonds

# Calculate probability
probability = red_kings / total_cards
percentage = probability * 100

print(f"\n Total cards in deck: {total_cards}")
print(f" Red Kings in deck: {red_kings}")
print(f"\n Probability = {red_kings}/{total_cards}")
print(f" Probability = {probability:.4f}")
print(f" Percentage = {percentage:.2f}%")

# Simplify the fraction
from fractions import Fraction
simplified = Fraction(red_kings, total_cards)
print(f" Simplified fraction: {simplified}")

print("\n Explanation:")
print("   Standard deck has 52 cards:")
print("   - 4 suits: Hearts ♥, Diamonds ♦, Spades ♠, Clubs ♣")
print("   - Each suit has 1 King")
print("   - Hearts and Diamonds are RED")
print("   - Spades and Clubs are BLACK")
print("   - Red Kings = King of Hearts + King of Diamonds = 2 cards")
print("   - Probability = 2/52 = 1/26 ≈ 3.85%")




print(f"| Bayes Theorem (Disease) | {p_disease_given_pos * 100:.1f}% chance of having virus |")
print(f"| Red King Probability | 1/26 = {percentage:.2f}% |")

print("\n" + "=" * 60)
print("✅ DAY 3 COMPLETE! Both Probability Problems Solved.")
print("=" * 60)
