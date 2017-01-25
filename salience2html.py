#!/usr/bin/env python

import sys

import numpy as np

CSS_HEADER = """span
{
display:inline-block;
}

span.A
{
color:red;
}

span.C
{
color:blue;
}

span.G
{
color:orange;
}

span.T
{
color:green;
}
"""

CSS_SEQ = """#S%s
{
font-size:100px;
font-family:"Arial Black";
letter-spacing:-60px;
transform:scale(0.2,%.3f);
vertical-align:-%spx;
}
"""

CSS_REF = """#R%s
{
font-size:100px;
font-family:"Arial Black";
letter-spacing:-60px;
transform:scale(0.2,0.2);
vertical-align:0px;
color:black;
margin-top:-77px;
}
"""

CSS_LINE = """hr
{
margin-top:-85px;
margin-left:7px;
width:%spx;
}
"""


def main():
    infile = open(sys.argv[1])
    
    line = infile.next().strip('\n')
    seq, salience_str = line.split('\t')
    salience_lst = map(float, salience_str.split(',')) 
        
    N = len(seq)
    salience_lst = np.array(salience_lst)
    for i in range(0, N):
        if salience_lst[i] < 0:
            salience_lst[i] = 0
    height_lst = salience_lst*1.0/salience_lst.max()
    
    print '<p class=\"SEQ\">'
    for i in range(0, N):
        print "<span class=\"%s\" id=\"S%s\">%s</span>" % (seq[i], i+1, seq[i])
    print '</p>'

    print '<hr>'

    print '<p class=\"REF\">'
    for i in range(0, N):
        print "<span class=\"%s\" id=\"R%s\">%s</span>" % (seq[i], i+1, seq[i])
    print '</p>'

    
    print '\n<style>'
    print CSS_HEADER
    print CSS_LINE % (N*21.5+10)
    
    for i in range(0, N):
        print CSS_SEQ % (i+1, 2*height_lst[i], 85*(1-height_lst[i]))

    for i in range(0, N):
        print CSS_REF % (i+1)
    
    
    print '</style>'
    
if __name__ == '__main__':
    main()

