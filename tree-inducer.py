__author__ = "Tristan Gaeta"

from math import log2
import sys

"""
This class will represent the tree structure.
"""
class Tree:
    def __init__(self):
        self.root = None

    def create(self,train,tune):
        issues = [i for i in range(len(train[0][2]))]
        p_d = 0 #calculate initial entropy
        for datum in train:
            if datum[1] == 'D': p_d += 1
        p_d /= len(train)
        self.root = self.__recurse__(train,issues,self.__entropy__(p_d,1-p_d),None)
        self.__size_tree__()
        self.__trim__(tune)

    def __entropy__(self,p1,p2):
        if p1 == 0 or p2 == 0:
            return 0
        e1 = -p1*log2(p1)
        e2 = -p2*log2(p2)
        return e1 + e2 #bits of entropy

    def __recurse__(self, data, issues, entropy, majority):
        majority = self.__majority__(data) or majority #keep track of most recent majority
        #base cases
        if len(data) == 0 or len(issues) == 0: 
            return majority 
        same_house = True   #if all representatives are in same house 
        same_record = True  #if all representatives have same record 
        for datum in data:
            if datum[1] != data[0][1]:
                same_house = False
            if datum[2] != data[0][2]:
                same_record = False
            if not same_record and not same_house:
                break
        if same_record or same_house:
            return majority
        #Find best split
        max_gain = float('-inf')
        child_entropies = None
        top_issue = None
        for issue in issues:
            #split data on issue 
            yea = {'D':0,'R':0}
            nay = {'D':0,'R':0}
            present = {'D':0,'R':0}
            for datum in data:
                party = datum[1]
                vote = datum[2][issue]
                if vote == "+":
                    yea[party] += 1
                elif vote == "-":
                    nay[party] += 1
                elif vote == ".":
                    present[party] += 1
                else: raise AssertionError("Unrecognized token: "+vote)
            #Calculate Gain
            yea_len = yea['D'] + yea['R']
            nay_len = nay['D'] + nay['R']
            pre_len = present['D'] + present['R']
            gain = entropy
            
            if yea_len != 0:            #yea
                yea['D'] /= yea_len
                yea['R'] /= yea_len
                e_y = self.__entropy__(yea['D'],yea['R'])
                gain -= yea_len/len(data) * e_y
            else:
                e_y = 0
            if nay_len != 0:            #nay
                nay['D'] /= nay_len
                nay['R'] /= nay_len
                e_n = self.__entropy__(nay['D'],nay['R'])
                gain -= nay_len/len(data) * e_n
            else:
                e_n = 0
            if pre_len != 0:            #present
                present['D'] /= pre_len
                present['R'] /= pre_len           
                e_p = self.__entropy__(present['D'],present['R'])
                gain -= pre_len/len(data) * e_p 
            else:
                e_p = 0
            #update best
            if gain > max_gain or (gain == max_gain and issue < top_issue):
                max_gain = gain
                child_entropies = (e_y,e_n,e_p)
                top_issue = issue
        #Split data
        yea = []
        nay = []
        present = []
        for datum in data:
            vote = datum[2][top_issue]
            if vote == "+":
                yea.append(datum)
            elif vote == "-":
                nay.append(datum)
            elif vote == ".":
                present.append(datum)
            else: raise AssertionError("Unrecognized token!")
        #recurse
        child_issues = [i for i in issues if i != top_issue]
        yea = self.__recurse__(yea,child_issues,child_entropies[0],majority)
        nay = self.__recurse__(nay,child_issues,child_entropies[1],majority)
        present = self.__recurse__(present,child_issues,child_entropies[2],majority)

        if yea == nay == present:   #if all same classification, turn into leaf
            return yea
        return Tree.Node(top_issue,yea,nay,present,majority)

    def classify(self,votes):
        def rec(node):
            if isinstance(node,str):
                return node
            if votes[node.issue] == '+':
                return rec(node.yea)
            if votes[node.issue] == '-':
                return rec(node.nay)
            if votes[node.issue] == '.':
                return rec(node.present)
            raise AssertionError("Unrecognized token: "+votes[node.issue])
        return rec(self.root)

    def __size_tree__(self):
        def rec(node):
            if isinstance(node,str):
                return 1
            size = 1 + rec(node.yea) + rec(node.nay) + rec(node.present)
            node.size = size
            return size
        return rec(self.root)

    def accuracy(self,data):
            correct = 0
            for datum in data:
                classification = self.classify(datum[2])
                if classification is None:
                    raise AssertionError("Broken tree: None classification")
                res = 1 if datum[1] == classification else 0
                correct += res
            return correct/len(data)

    def __trim__(self,data):
        def find_best_snip(node,best):
            if isinstance(node,str):
                return best
            personal_best = None
            personal_acc = -1
            personal_snips = 0
            #Find personal best for node
            yea = node.yea
            if not isinstance(yea,str):
                node.yea = yea.majority
                acc = self.accuracy(data)
                node.yea = yea
                if acc > personal_acc or (acc == personal_acc and node.yea.size > personal_snips):
                    personal_best = (node, 'yea')
                    personal_acc = acc
                    personal_snips = node.yea.size
            nay = node.nay
            if not isinstance(nay,str):
                node.nay = nay.majority
                acc = self.accuracy(data)
                node.nay = nay
                if acc > personal_acc or (acc == personal_acc and node.nay.size > personal_snips):
                    personal_best = (node, 'nay')
                    personal_acc = acc
                    personal_snips = node.nay.size
            present = node.present
            if not isinstance(present,str):
                node.present = present.majority
                acc = self.accuracy(data)
                node.present = present
                if acc > personal_acc or (acc == personal_acc and node.present.size > personal_snips):
                    personal_best = (node, 'present')
                    personal_acc = acc
                    personal_snips = node.present.size
            #update overall best
            def get_best(best,personal):
                if personal[1] > best[1] or (personal[1] == best[1] and personal[2] > best[2]):
                    return personal
                return best
            best = get_best(best,(personal_best,personal_acc,personal_snips))
            #recurse
            r1 = find_best_snip(node.yea,best)
            best = get_best(best,r1)
            r2 = find_best_snip(node.nay,best)
            best = get_best(best,r2)
            r3 = find_best_snip(node.present,best)
            best = get_best(best,r3)
            return best
        #begin algo
        acc = self.accuracy(data)
        trim = find_best_snip(self.root,(None,acc,0))
        
        #Snip if better
        if trim[0] is not None:
            if trim[0][1] == 'yea':
                trim[0][0].yea = trim[0][0].yea.majority
            elif trim[0][1] == 'nay':
                trim[0][0].nay = trim[0][0].nay.majority
            elif trim[0][1] == 'present':
                trim[0][0].present = trim[0][0].present.majority
            self.__trim__(data)

    def __majority__(self,data):
        dems = 0
        reps = 0
        for datum in data:
            if datum[1] == 'D':
                dems += 1
            elif datum[1] == 'R':
                reps += 1
            else: raise AssertionError('Unknown party: '+datum[1])
        if dems == reps:
            return None
        return 'D' if dems > reps else 'R'

    def print_tree(self):
        def rec(node, depth, label):
            if isinstance(node,str):
                print('  '*depth+label+node)
            else:
                print('  '*depth+label+"Issue "+chr(65+node.issue)+':')
                rec(node.yea,depth+1,'+ ')
                rec(node.nay,depth+1,'- ')
                rec(node.present,depth+1,'. ')
        rec(self.root,0,'')

    """
    This class represents a node of the decision tree. Each node has three children
    """
    class Node:
        def __init__(self,issue, yea, nay, present, majority):
            self.issue = issue
            self.yea = yea
            self.nay = nay
            self.present = present
            self.majority = majority
            self.size = None

def cross_validate(data):
    num_correct = 0
    for i in range(len(data)):
        train, tune = split_data(data,i)
        tree = Tree()
        tree.create(train, tune)
        if tree.classify(data[i][2]) == data[i][1]:
            num_correct += 1
    acc = num_correct*100/len(data)
    print('Accuracy: %.3f' % acc,"%")

def split_data(data,leave_out):
    train = []
    tune = []
    i = 0 
    off = 0
    for datum in data:
        if i != leave_out:
            if (i - off) % 4 == 0:
                tune.append(datum)
            else:
                train.append(datum)
        else: off = 1
        i += 1
    return (train, tune)

if __name__ == "__main__":
    file = open(sys.argv[1],'r')
    data = []
    for line in file.readlines():
        member_id, party, votes = line.split('\t')
        votes = votes.removesuffix('\n')
        data.append((member_id,party,votes))
    file.close()
    train, tune = split_data(data,None)
    tree = Tree()
    tree.create(train, tune)
    tree.print_tree()
    cross_validate(data)