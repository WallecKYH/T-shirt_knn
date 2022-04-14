# importing module
import pandas
import sns as sns
from pandas import *
import csv
from itertools import zip_longest
from math import sqrt
import numpy as np
from csv import writer
from csv import reader



data = read_csv("male.csv")

# converting column data to list
Height = data['Heightin'].tolist()
Weight = data['Weightlbs'].tolist()
Chest= data['chestcircumference'].tolist()
Waist= data['waistcircumference'].tolist()

# printing list data
print('Heightin:', Height)
print('Weightlbs:',Weight)
print('chestcircumference:',Chest)
print('waistcircumference:',Waist)

header = [ 'Heightin', 'Weightlbs','chestcircumference','waistcircumference']

data2 = [
]

columns_data = zip_longest(*data2)

list_1 = Height
list_2 = Weight
list_3 = Chest
list_4 = Waist
list_5 = []

with open('T-Shirt.csv', 'w', newline='') as f:
    writer = csv.writer(f)

    for w in range(len(list_1)):


            writer.writerows(data)

    with open('male.csv', 'r') as file:
        reader = csv.reader(file)

        for row in reader:
            print(row)


    writer.writerow([list_1[w], list_2[w], list_3[w], list_4[w], list_5[w]])

    f.close()

df = pandas.read_csv("T-Shirt.csv")
df["Size"] = "Empty"
df.to_csv("T-Shirt.csv", index=False)





#height = input("Enter your height: ")
#weight = input("Enter ur weight in lbs: ")

# Display all values on screen
#print("\n")
#print("Ur height and weight are ")
#print(height, weight)
#print("Then ur T-shirt size should be ")


# calculate the Euclidean distance between two vectors
#def euclidean_distance(row1, row2):
 #   distance = 0.0
  #  for i in range(len(row1) - 1):
   #     distance += (row1[i] - row2[i]) ** 2
    #return sqrt(distance)


