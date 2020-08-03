
ip = "192.168.1.2"

def convertIPAddress(ip, base=256):
    toReturn = 0.0
    list = ip.split(".")
    length = len(list)
    for i in range(length):
        added = float(list[length-i-1])*(base**i)
        toReturn += added
        print(toReturn)
    return toReturn

print(type(convertIPAddress(ip)))
