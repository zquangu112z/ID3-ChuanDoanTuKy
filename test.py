import TreeDecision

def main():
    #Insert input file
    """
    IMPORTANT: Change this file path to change training data
    """
    file = open('data.csv')
    """
    IMPORTANT: Change this variable too change target attribute
    """
    target = 'result'
    data = [[]]
    for line in file:
        line = line.strip("\r\n")
        data.append(line.split(','))
    data.remove([])
    attributes = data[0]
    print attributes
    data.remove(attributes)
    #Run ID3
    # Run ID3
    tree = TreeDecision.makeTree(data, attributes, target)
    print "generated decision tree"
    # Generate program
    file = open('result.txt', 'w')

    file.write("tree = %s\n" % str(tree))

    print "written program"

if __name__ == '__main__':
    main()