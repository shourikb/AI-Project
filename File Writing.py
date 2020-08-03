import os
from StringToFloat import convertIPAddress


with open("EmergeSyncString.csv", "r") as reader:
    #row_count = sum(1 for row in reader)
    #print(row_count)
    #print(reader.readline())
    f = open("C://Users/Shourik/PycharmProjects/AI/New.txt", "w")
    line = reader.readline()
    line = reader.readline()
    while line:
        print("Line", line)
        content_list = line.split(',')
        print("Content List", content_list)
        for j in range(5):
            if j < 2:
                print("j=", j)
                print("content_list[j] = ", content_list[j])
                f.write(content_list[j] + ",")
        f.write("\n")
        line = reader.readline()
    f.close()



