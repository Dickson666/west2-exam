import random

Tp = ["","Spade","Heart","Diamond","Club"]

tP = ['','A','2','3','4','5','6','7','8','9','10','J','Q','K']

a = [[]for i in range(54)]

for i in range(1, 14):
    for j in range (1, 5): 
        a[(i - 1) * 4 + j - 1] = [Tp[j], tP[i]]
        # print(Tp[j])

a[52] = ["", "JOKER"]
a[53] = ["", "joker"]

random.shuffle(a)

for i in range(0, 3):
    with open("player" + str(i + 1) + ".txt", "w") as f:
        for j in range(0 + i * 17, 17 + i * 17):
            if a[j][0] == "" and (a[j][1] == "JOKER" or a[j][1] == 'joker'):
                f.write(a[j][1] + '\n')
            # print(j)
            else :
                b = a[j][0] + " " + a[j][1] + '\n'
                f.write(b)

with open("other.txt", "w") as f:
    for i in range(51, 54):
        b = a[i][0] + a[i][1] + '\n'
        f.write(b)
