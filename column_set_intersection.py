import csv

# Initialize two empty sets to store the unique values of each column
first_column_set = set()
second_column_set = set()

# Open the CSV file
with open('System Call Categorization - Sheet2.csv', mode='r', newline='') as file:
    # Create a CSV reader object
    csv_reader = csv.reader(file)

    # Iterate over each row in the CSV file
    for row in csv_reader:
        # Add the values from the first and second columns to their respective sets
        if len(row) >= 2:
            first_column_set.add(row[0])
            second_column_set.add(row[1])

# Print the sets containing the first and second columns
#print("First Column Set:", first_column_set)
#print("Second Column Set:", second_column_set)

intersection_set = first_column_set & second_column_set
intersection_count = len(intersection_set)

print("Intersection Set:", intersection_set)
print("Number of elements in the Intersection Set:", intersection_count)

