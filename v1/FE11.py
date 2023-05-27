import math
import random

inputList = {
        'a': lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: x1,
        'b': lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: x2,
        'c': lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: x3,
        'd': lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: x4,
        'e': lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: x5,
        'f': lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: x6,
        'g': lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: x7,
        'h': lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: x8,
        'i': lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: x9,
        'j': lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: x10,
        'k': lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: x11
              }

inputs = 'abcdefghijk'
operators = '+*/'
activations = 'SR'

def activate(f,acfun):
    if acfun == 'S':
        return lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: 0.0 if f(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)<-20 else 1.0 if f(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)>20 else 1/(1+math.exp(-f(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)))
    elif acfun == 'R':
        return lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: 0.0 if f(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11) < 0 else f(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)

def compose(f,g,operator):
    if operator == '*':
        return lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11:f(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)*g(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)
    elif operator == '+':
        return lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11:f(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)+g(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)
    elif operator == '/':
        return lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11:math.copysign(math.inf,f(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)) if g(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11) == 0.0 else f(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)/g(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11)

def getNextSeg(gene,head):
    pCount = 0
    comp = False
    for i in range(head,len(gene)):
        if pCount > 1:
            comp = True
        if gene[i] == '(':
            pCount += 1
        elif gene[i] == ')':
            pCount -= 1
        if pCount == 0:
            return gene[head+1:i], comp

def getKernel(gene):
    ind = gene.find('(')
    if ind == -1:
        return gene
    else:
        return gene[:ind]
        
def str2lam(ele):
    if len(ele) == 1:
        return inputList[ele]
    else:
        return lambda x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11: float(ele)

def Compile(gene):
    kernel = getKernel(gene)
    f = str2lam(kernel)
    head = len(kernel)
    while head<len(gene):
        seg,comp = getNextSeg(gene,head)
        if comp:
            f = compose(f,Compile(seg[1:]),seg[0])
        elif seg[0] in activations:
            f = activate(f,seg[0])
        else:
            f = compose(f,str2lam(seg[1:]),seg[0])
        head += len(seg)+2
    return f

def decision(P):
    p = random.random()
    if p<P:
        return True
    else:
        return False

def substitute(Type,item,p):
    if decision(p):
        if Type == 'element':
            if decision(0.5):
                return random.choice(inputs.replace(item,''))
            elif item in inputs:
                return "{:.8f}".format(random.gauss(0.0,1.0))
            else:
                return item
        elif Type == 'operator':
            return random.choice(operators.replace(item,''))
        else:
            return random.choice(activations.replace(item,''))
    else:
        return item
        

def generateSeg():
    if decision(0.05):
        return '('+random.choice(activations)+')'
    elif decision(0.5):
        return '('+random.choice(operators)+random.choice(inputs)+')'
    else:
        return '('+random.choice(operators)+"{:.8f}".format(random.gauss(0.0,1.0))+')'

def mutate(gene,p,t):
    newGene = ''
    kernel = getKernel(gene)
    newGene = substitute('element',kernel,p)
    head = len(kernel)
    if decision(p):
        newGene = newGene + generateSeg()
    while head<len(gene):
        seg,comp = getNextSeg(gene,head)
        if decision(p) and t:
            head += len(seg)+2
            continue
        newGene = newGene +'('
        if comp:
            newGene = newGene + substitute('operator',seg[0],p)
            newGene = newGene + mutate(seg[1:],p,t)
        else:
            if seg[0] in operators:
                newGene = newGene + substitute('operator',seg[0],p)
                newGene = newGene + substitute('element',seg[1:],p)
                if decision(p):
                    newGene = newGene + generateSeg()
            else:
                newGene = newGene + substitute('activation',seg[0],p)
        newGene = newGene +')'
        if decision(p):
                    newGene = newGene + generateSeg()
        head += len(seg)+2
    return newGene

def substituteNum(item,sigma):
    newItem = item
    if item[0] in '-0123456789':
        mu = float(item)
        newItem = "{:.8f}".format(random.gauss(mu, sigma))
    return newItem
    
def para_learning(gene,sigma):
    newGene = ''
    kernel = getKernel(gene)
    newGene = substituteNum(kernel,sigma)
    head = len(kernel)
    while head<len(gene):
        seg,comp = getNextSeg(gene,head)
        newGene = newGene +'('
        if comp:
            newGene = newGene + seg[0] + para_learning(seg[1:],sigma)
        else:
            if seg[0] in operators:
                newGene = newGene + seg[0] + substituteNum(seg[1:],sigma)
            else:
                newGene = newGene + seg
        newGene = newGene +')'
        head += len(seg)+2
    return newGene
