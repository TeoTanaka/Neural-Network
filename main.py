training = ["hello","how","are","you","I","am","doing","well"]
training_dict = {}
#N GRAM LANGUAGE MODEL

#BREAK APART training and itemize each word with the word that comes before it
for a in training:
    if training.index(a) > 0: #for every word after index 0
        training_dict[a] = training[training.index(a)-1]

#training dictionary is every letter : the letter before itself

#GETTING INPUT
print("PARROT: Hello there! How can I help you today?")
inp = input("YOU: ")
inp_split = inp.split()
last_inp = inp_split[-1]#last letter of the input

#IF the last letter of the input == to something in the training data

for a in training:
    if a == 