import os
from StringToFloat import convertIPAddress
import random


def writeCSV(file, name, columns=7, anom_percent=66):
    proto_dict = {"RSYNC": "0.1", "TCP": "0.2", "IGMPv6": "0.3", "IGMPv3": "0.4"}
    with open(file, "r") as reader:
        #print(row_count)
        #print(reader.readline())
        f = open("C://Users/Shourik/PycharmProjects/AI/" + name + ".csv", "w")
        f.write("2," + str(columns) + ", anomaly, normal" + "\n")
        line = reader.readline()
        line = reader.readline()
        while line:
            isError = False
            content_list = line.split('\t')
            anom = random.randint(1, 100)
            if anom<anom_percent:
                for j in range(columns-1):
                    if j >= len(content_list):
                        f.write("0.5,")
                        isError = True
                        continue
                    try:
                        #print("Content list for integers = " + content_list[j])
                        f.write(str(float(content_list[j])) + ",")
                    except ValueError:
                        try:
                            toWrite = convertIPAddress(content_list[j])
                            f.write(str(toWrite) + ",")
                        except ValueError:
                            w = proto_dict.get(content_list[j])
                            if(w==None):
                                f.write("0.5,")
                                isError = True
                            else:
                                f.write(w + ",")
                if isError:
                    f.write("0")
                else:
                    f.write("1")
            else:
                toAnom1 = random.randint(0,6)
                toAnom2 = random.randint(0,6)
                for j in range(columns-1):
                    if j >= len(content_list):
                        f.write("0.5,")
                        continue
                    elif j == toAnom1 or j == toAnom2:
                        bad = random.random() * 12
                        # print(bad)
                        f.write(str(bad) + ",")
                        continue
                    try:
                        # print("Content list for integers = " + content_list[j])
                        f.write(str(float(content_list[j])) + ",")
                    except ValueError:
                        try:
                            toWrite = convertIPAddress(content_list[j])
                            f.write(str(toWrite) + ",")
                        except ValueError:
                            w = proto_dict.get(content_list[j])
                            if (w == None):
                                f.write("0.5,")
                            else:
                                f.write(w + ",")
                f.write("0")
            f.write("\n")
            isError = False
            line = reader.readline()
        f.close()

def readFileToList(file):
    toReturn = []
    with open(file, "r") as reader:
        reader.readline()
        line = reader.readline()
        content_list = line.split(",")
        while line:
            content_list.remove(content_list[len(content_list)-1])
            for i in range(len(content_list)):
                content_list[i] = float(content_list[i])
            toReturn.append(content_list)
            line = reader.readline()
            content_list = line.split(",")
    return toReturn


print(readFileToList("Predict.csv"))