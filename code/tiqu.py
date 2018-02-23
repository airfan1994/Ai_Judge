# -*- coding:utf-8 -*- 
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

shuzi = ['一','二','三','四','五','六','七','八','九']
fanti_shuzi = ['壹','贰','叁','肆','伍','陆','柒','捌','玖']
shuzi_weight = [1,2,3,4,5,6,7,8,9]
danwei= ['十','百','千','万']
fanti_danwei = ['拾','佰','仟','万']
danwei_weight = [10,100,1000,10000]
def is_number(s):
    try:
        p = float(s)            
        return True
    except ValueError:
        pass
    
    return False

def jiexi_money(last_word,word):
    if '亿' in last_word or '亿' in word:
        return 500000000
    if '千万' in last_word or '千万' in word:
        return 50000000
    if '百万' in last_word or '百万' in word:
        return 5000000
    if '十万' in last_word or '十万' in word:
        return 600000
    if is_number(last_word):
        first_money = float(last_word)
    else:
        first_money = 0
        cur_money = 1
        for ss in last_word:
            if ss in shuzi:
                cur_money = shuzi_weight[shuzi.index(ss)]
            elif ss in fanti_shuzi:
                cur_money = shuzi_weight[fanti_shuzi.index(ss)]
            elif ss in danwei:
                first_money +=cur_money * danwei_weight[danwei.index(ss)]
                cur_money = 0
            elif ss in fanti_danwei:
                first_money +=cur_money * danwei_weight[fanti_danwei.index(ss)]
                cur_money = 0
    sum_money = 0
    cur_money = first_money
    base = 1
    for ss in word:
        if ss in shuzi:
                cur_money = shuzi_weight[shuzi.index(ss)]
        elif ss in fanti_shuzi:
                cur_money = shuzi_weight[fanti_shuzi.index(ss)]
        elif ss in danwei:
            sum_money +=cur_money * danwei_weight[danwei.index(ss)]
            cur_money = 0
        elif ss in fanti_danwei:
            sum_money +=cur_money * danwei_weight[fanti_danwei.index(ss)]
            cur_money = 0
        elif ss == '美':
            base = 7
        elif ss == '日':
            base =0.0586
    sum_money +=cur_money
    return sum_money * base


if __name__=='__main__':
    f=open('train_2.txt')
    f2 = open('trainjine.txt','w')
    line = f.readline()
    while len(line)>0:
        if len(line)>10:
            last_word = ''
            total = 0
            word_list = line.split(',')
            for word in word_list:
                if '元' in word:
                    total+=jiexi_money(last_word,word)
                last_word = word
            f2.write(str(total)+"\n")     
        line = f.readline()
    f.close()
    f2.close()
    f=open('test_2.txt')
    f2 = open('testjine.txt','w')
    line = f.readline()
    while len(line)>0:
        if len(line)>10:
            last_word = ''
            total = 0
            word_list = line.split(',')
            for word in word_list:
                if '元' in word:
                    total+=jiexi_money(last_word,word)
                last_word = word
            f2.write(str(total)+"\n")     
        line = f.readline()
    f.close()
    #print	jiexi_money('贰千','余万元')
    #print 	jiexi_money('2000','美元')
