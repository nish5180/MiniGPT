import csv
import random

# Sample fragments for encoder texts (banana facts)
banana_facts = [
    "are naturally curved",
    "are rich in potassium",
    "grow in hanging clusters",
    "are botanically classified as berries",
    "are used in many culinary dishes",
    "ripen best in tropical climates",
    "are among the world's most popular fruits",
    "come in many varieties",
    "contain essential dietary fiber",
    "have a unique, creamy texture"
]

# Expanded list of Shakespeare-inspired templates for decoder target texts.
shakespeare_templates = [
    "O, gentle fruit, thy {fact} doth mesmerize mine eyes and kindle a fiery delight.",
    "Lo! In thy {fact} lies a beauty most divine, as if nature herself were a poet.",
    "Thou art a wondrous fruit, for thy {fact} doth charm the very spirit of the morn.",
    "Behold, fair banana, with thy {fact} thou dost outshine all mortal produce, a marvel most rare.",
    "Ah, sweet delight, in thy {fact} is found a splendor that doth stir the soul to rapture.",
    "Hark! Thy {fact} sings to the heart as soft whispers of an ancient bard.",
    "Verily, thy {fact} doth render thee a celestial wonder amid earthly fare.",
    "Thou art a fruit of wonder, whose {fact} inspires verses steeped in timeless grace.",
    "Mark well, for thy {fact} is as enchanting as a midsummer night's dream.",
    "With thy {fact} thou dost grace our table, a banquet for both mind and heart.",
    "In thy {fact} lies a secret, subtle as the whispers of yonder winds.",
    "O marvelous banana, thy {fact} is a sonnet penned by the hand of the stars.",
    "As the dawn unfolds, so does thy {fact}, gentle fruit, in a display most profound.",
    "Thy {fact}, dear banana, bestows upon thee a charm that doth rival the very muse.",
    "Attend! For in thy {fact} there echoes a tale of wonder, as if told by fate herself."
]

# Path to the CSV file where the dataset will be written.
output_file = "big_dataset.csv"

# Number of rows in the generated dataset.
num_rows = 100

# Write the CSV file.
with open(output_file, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    # Write header row
    writer.writerow(["encoder_input", "decoder_target"])
    
    for i in range(num_rows):
        # Choose a random fact from banana_facts
        fact = random.choice(banana_facts)
        # Create the encoder input text.
        encoder_text = f"Banana facts: Bananas {fact}, and they flourish under the warmest sun."
        # Choose a random Shakespeare-inspired template and fill it in with the chosen fact.
        template = random.choice(shakespeare_templates)
        decoder_text = template.format(fact=fact)
        # Write the paired record to the CSV file.
        writer.writerow([encoder_text, decoder_text])

print(f"Dataset with {num_rows} rows created in {output_file}.")
