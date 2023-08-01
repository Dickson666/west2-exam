ls = [[1 for i in range (10)] for i in range(5)]
for i in range(5) :
    for j in range(10):
        print(ls[i][j], end = " ")
    print("")

print("-----------------")

rs = [[ls[i][j] for i in range(5)]for j in range(10)]
for i in range(10) :
    for j in range(5):
        print(rs[i][j], end = " ")
    print("")
