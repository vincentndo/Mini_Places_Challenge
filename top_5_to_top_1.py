import csv


FILENAME = "top_5_predictions.test.csv"
NEWFILE = "top_1_predictions.test.csv"

buff = []
with open(FILENAME) as file, open(NEWFILE, 'w') as newfile:
    lines = csv.reader(file)
    writer = csv.writer(newfile)
    for line in lines:
        writer.writerow(line[:2])
