def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:                              #全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374): #全角字符（除空格）根据关系转化
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring
def main():
    labelFile = "/home/public/yushilin/ocr/data/train.txt"
    dstFile = "/home/public/yushilin/ocr/data/train1.txt"
    writedata = open(dstFile,"w")
    with open(labelFile,'r') as f:
        data = f.readlines()
    for line in data:
        splitLine = line.split("\t")
        label = splitLine[3]
        label = strQ2B(label)
        finalLine = splitLine[0]+"\t"+splitLine[1]+"\t"+splitLine[2]+"\t"+label
        writedata.write(finalLine)
    writedata.close()
if __name__=="__main__":
    main()