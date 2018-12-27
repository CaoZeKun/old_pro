# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 20:36:50 2018

@author: 四级必过
"""
import csv
import os
import getList as sg
import re
def getQuoteList():                                              #获取股票报价列表
    html = open('data.html','rb').read().decode('gb2312','ignore')     #以二进制格式读HTML股票代码
    #print (html)
    pattern = re.compile( r'\((\d*)\)', re.S )                  #正则表达式r（界位符），re.S不区分大小写
    return re.findall( pattern, html ) 
#查询函数：从资产负债表、利润表、现金流量表中读取属性的值
def  find_value(path,year,item):                     #path:资产负债表保存路径，year报表年份，item：资产负债表项目
    yearstr = str(year) + '-12-31'
  #  choose = open(path, "r")                        #判断是否存在该年份年度报表
 #   yearlist = choose.readline()
    try:
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            column_yearline = [row[yearstr] for row in reader]
        f.close()
        with open(path, "r") as f:
            reader = csv.DictReader(f)
            column_itemline = [row['报告日期'] for row in reader]
        f.close()
        index = column_itemline.index(item)
        value = column_yearline[index]
        if value == '--':
            return  0
        else:
            return  value
    except Exception as err:
       # print(err)
        return -1
            

#保留浮点数小数点后四位
def  retain(f):                                       #参数f指浮点数
    f = f*10000
    return  round(f)/10000

#净资产收益率             计算公式:    净利润/所有者权益(或股东权益)合计*100%
def net_assets_income_rate(balance_path,profit_path,year):
    b = float(find_value(profit_path,year,'净利润(万元)'))
    c = float(find_value(balance_path,year,'所有者权益(或股东权益)合计(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#利润率                   计算公式:    净利润/营业收入*100%
def profit_rate(profit_path,year):
    b = float(find_value(profit_path,year,'净利润(万元)'))
    c = float(find_value(profit_path,year,'营业收入(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#主营业务收入增长率        计算公式:    (本期营业收入-上期营业收入)/上期营业收入*100%
def main_business_growth_rate(profit_path,year):
    prior_year = year - 1
    b = float(find_value(profit_path,year,'营业收入(万元)')) - float(find_value(profit_path,prior_year,'营业收入(万元)'))
    c = float(find_value(profit_path,prior_year,'营业收入(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#净利润增长率             计算公式:     (本期净利润-上期净利润)/上期净利润*100%
def net_profit_growth_rate(profit_path,year):
    prior_year = year - 1
    b = float(find_value(profit_path,year,'净利润(万元)')) - float(find_value(profit_path,prior_year,'净利润(万元)'))
    c = float(find_value(profit_path,prior_year,'净利润(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#总资产增长率             计算公式:     (本期资产总计-上期资产总计)/上期资产总计*100%
def total_assets_growth_rate(balance_path,year):
    prior_year = year - 1
    b = float(find_value(balance_path,year,'资产总计(万元)')) - float(find_value(balance_path,prior_year,'资产总计(万元)'))
    c = float(find_value(balance_path,prior_year,'资产总计(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#净资产增长率             计算公式:     (本期所有者权益(或股东权益)合计-上期所有者权益(或股东权益)合计)
#                                                /上期所有者权益(或股东权益)合计*100%
def net_assets_growth_rate(balance_path,year):
    prior_year = year - 1
    b = float(find_value(balance_path,year,'所有者权益(或股东权益)合计(万元)')) - float(find_value(balance_path,prior_year,'所有者权益(或股东权益)合计(万元)'))
    c = float(find_value(balance_path,prior_year,'所有者权益(或股东权益)合计(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#每股收益增长率           计算公式:     (本期基本每股收益-上期基本每股收益)/(上期基本每股收益)*100%
def earnings_per_share_growth_rate(profit_path,year):
    prior_year = year - 1
    b = float(find_value(profit_path,year,'基本每股收益')) - float(find_value(profit_path,prior_year,'基本每股收益'))
    c = float(find_value(profit_path,prior_year,'基本每股收益'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#总资产周转率             计算公式:    营业收入*2/（本期所有者权益(或股东权益)合计+上期所有者权益(或股东权益)合计）
def total_assets_turnover(balance_path,profit_path,year):
    prior_year = year - 1
    b = float(find_value(profit_path,year,'营业收入(万元)')) * 2
    c = float(find_value(balance_path,year,'所有者权益(或股东权益)合计(万元)')) + float(find_value(balance_path,prior_year,'所有者权益(或股东权益)合计(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#净资产周转率             计算公式:    营业收入*2/（本期资产总计+上期资产总计）*100%
def net_assets_turnover(balance_path,profit_path,year):
    prior_year = year - 1
    b = float(find_value(profit_path,year,'营业收入(万元)')) * 2
    c = float(find_value(balance_path,year,'资产总计(万元)')) + float(find_value(balance_path,prior_year,'资产总计(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#存货周转率               计算公式:    营业收入*2/（本期存货+上期存货）*100%
def stock_turnover(balance_path,profit_path,year):
    prior_year = year - 1
    b = float(find_value(profit_path,year,'营业收入(万元)')) * 2
    c = float(find_value(balance_path,year,'存货(万元)')) + float(find_value(balance_path,prior_year,'存货(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#应收帐款周转率           计算公式:    营业收入*2/（本期应收账款+上期应收账款）*100%
def accounts_receivable_turnover(balance_path,profit_path,year):
    prior_year = year - 1
    b = float(find_value(profit_path,year,'营业收入(万元)')) * 2
    c = float(find_value(balance_path,year,'应收账款(万元)')) + float(find_value(balance_path,prior_year,'应收账款(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#总资产收益率             计算公式:    净利润*2/（本期资产总计+上期资产总计）×100%
def return_on_total_assets(balance_path,profit_path,year):
    prior_year = year - 1
    b = float(find_value(profit_path,year,'净利润(万元)')) * 2
    c = float(find_value(balance_path,year,'资产总计(万元)')) + float(find_value(balance_path,prior_year,'资产总计(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#净利润率                 计算公式:    （净利润/营业收入）×100%
def net_profit_rate(profit_path,year):
    b = float(find_value(profit_path,year,'净利润(万元)'))
    c = float(find_value(profit_path,year,'营业收入(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#毛利率                   计算公式:    （营业收入-营业成本）/营业收入*100%
def gross_profit_rate(profit_path,year):
    b = float(find_value(profit_path,year,'营业收入(万元)')) - float(find_value(profit_path,year,'营业成本(万元)'))
    c = float(find_value(profit_path,year,'营业收入(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#资产负债率               计算公式:    负债合计/资产总计*100%
def asset_liability_ratio(balance_path,year):
    b = float(find_value(balance_path,year,'负债合计(万元)'))
    c = float(find_value(balance_path,year,'资产总计(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#流动比率                 计算公式:    流动资产合计/流动负债合计
def current_ratio(balance_path,year):
    b = float(find_value(balance_path,year,'流动资产合计(万元)'))
    c = float(find_value(balance_path,year,'流动负债合计(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value

#速动比率                 计算公式:    （流动资产合计-存货）/流动负债合计
def quick_ratio(balance_path,year):
    b = float(find_value(balance_path,year,'流动资产合计(万元)')) - float(find_value(balance_path,year,'存货(万元)'))
    c = float(find_value(balance_path,year,'流动负债合计(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#长期资产适合率           计算公式:    （所有者权益(或股东权益)合计+非流动负债合计）/（资产总计+长期股权投资）*100%
def Long_term_asset_suitability(balance_path,year):
    b = float(find_value(balance_path,year,'所有者权益(或股东权益)合计(万元)')) + float(find_value(balance_path,year,'非流动负债合计(万元)'))
    c = float(find_value(balance_path,year,'资产总计(万元)')) + float(find_value(balance_path,year,'长期股权投资(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#利息保障倍数             计算公式:    （利润总额+财务费用）/利息支出
def time_interest_earned_ratio(profit_path,year):
    b = float(find_value(profit_path,year,'利润总额(万元)')) + float(find_value(profit_path,year,'财务费用(万元)'))
    c = float(find_value(profit_path,year,'利息支出(万元)'))
    if  b == -1  or c == -1 or c == 0:
        return  -1
    else:
        value = b/c
        value = retain(value)
        return value
#经营现金流与负债比       计算公式:    经营活动产生现金流量净额/流动负债
def operating_cash_flow_and_liability_ratio(balance_path,cashflow_path,year):
    yearstr = str(year) + '-12-31'
    try:
        with open(cashflow_path, "r") as f:
            reader = csv.DictReader(f)
            column_name = [row[yearstr] for row in reader]
        f.close()
        if len(column_name) < 80:
            print('文件有误')
            return  -1
        else:
            b = float(column_name[80])
            if b == '--':
                b = 0
            c = float(find_value(balance_path,year,'流动负债合计(万元)'))
            if  b == -1  or c == -1 or c == 0:
                return  -1
            else:
                if b=='--' or c == '--':
                    return -1
                else:
                    value = float(b)/c
                    value = retain(value)
                    return value
    except Exception as err:
        #print(err)
        return -1
           
    
#净利润现金含金量         计算公式:    净利润/现金及现金等价物的净增加额
def net_profit_cash_amount(profit_path,cashflow_path,year):
    yearstr = str(year) + '-12-31'
    try:
        with open(cashflow_path, "r") as f:
            reader = csv.DictReader(f)
            column_name = [row[yearstr] for row in reader]
        f.close()
        c = float(column_name[88])
        if c == '--':
            return -1
        b = float(find_value(profit_path,year,'净利润(万元)'))
    #c = float(find_value(cashflow_path,year,'现金及现金等价物的净增加额(万元)'))
        if  b == -1  or c == -1 or c == 0:
            return  -1
        else:
            value = b/c
            value = retain(value)
            return value
    except Exception as err:
       # print(err)
        return -1



#计算财务报表指标
def calculate(balance_path,profit_path,cashflow_path):
    debt = open(balance_path , "r")
    debtlist = debt.readline()
    debt.close()
    yearlist = []
    year = re.findall('[\w()-]{2,200}',debtlist)  
    del year[0]
    for x in range(len(year)):
        if str(year[x][5]) == '1':
            yearlist.append(int(year[x][0]) *1000+int(year[x][1])*100+int(year[x][2])*10+int(year[x][3]))
  #  print(yearlist)
    
    
    
    #print(year)
    for x in range(len(yearlist)):
        
        v = net_assets_income_rate(balance_path,profit_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
                print('净资产收益率',format(v,'10.2%'))
    
    
        v = profit_rate(profit_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('利润率',format(v,'10.2%'))
        
        v = main_business_growth_rate(profit_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('主营业务收入增长率',format(v,'10.2%'))
        
        v = net_profit_growth_rate(profit_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('净利润增长率',format(v,'10.2%'))
        
        v = total_assets_growth_rate(balance_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('总资产增长率',format(v,'10.2%'))
        
        v = net_assets_growth_rate(balance_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('净资产增长率',format(v,'10.2%'))
        
        v = earnings_per_share_growth_rate(profit_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('每股收益增长率',format(v,'10.2%'))
        
        v = total_assets_turnover(balance_path,profit_path,yearlist[x])
        if  v==-1 or v==0:
            print('--')
        else:
            print('总资产周转率',format(v,'10.2%'))
        
        v = net_assets_turnover(balance_path,profit_path,yearlist[x])
        if  v==-1 or v==0:
            print('--')
        else:
            print('净资产周转率',format(v,'10.2%'))
        
        v = stock_turnover(balance_path,profit_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('存货周转率',format(v,'10.2%'))
    
  
        v = accounts_receivable_turnover(balance_path,profit_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('应收帐款周转率',format(v,'10.2%'))
 
  
        v = return_on_total_assets(balance_path,profit_path,yearlist[x])
        if  v==-1 or v==0:
            print('--')
        else:
            print('总资产收益率',format(v,'10.2%'))
   
        v = net_profit_rate(profit_path,yearlist[x])
        if  v==-1 or v==0:
            print('--')
        else:
            print('净利润率',format(v,'10.2%'))
        
        v = gross_profit_rate(profit_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('毛利率',format(v,'10.2%'))
        
        
        v = asset_liability_ratio(balance_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('资产负债率',format(v,'10.2%'))
    
        v = current_ratio(balance_path,yearlist[x])
        if  v==-1 or v==0:
            print('--')
        else:
            print('流动比率',v)
     
        v = quick_ratio(balance_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('速动比率',v)
      
    
        v = Long_term_asset_suitability(balance_path,yearlist[x])
        if  v==-1 or v==0:
            print('--')
        else:
            print('长期资产适合率:',format(v,'10.2%'))
        
        v = time_interest_earned_ratio(profit_path,yearlist[x])
        if  v==-1 or v==0:
            print('--')
        else:
            print('利息保障倍数:',format(v,'10.2%'))
   
     
        v = operating_cash_flow_and_liability_ratio(balance_path,cashflow_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('经营现金流与负债比:',v)
        
        v = net_profit_cash_amount(profit_path,cashflow_path,yearlist[x])
        if  v==-1:
            print('--')
        else:
            print('净利润现金含金量：',v)    
 


def write(balance_path,profit_path,cashflow_path,write_path,s):
    debt = open(balance_path , "r")
    debtlist = debt.readline()
    debt.close()
    yearlist = []
    year = re.findall('[\w()-]{2,200}',debtlist)  
    if len(year) > 0:
        del year[0]
    for x in range(len(year)):
        if len(year[x]) > 5:
            if str(year[x][5]) == '1':
                yearlist.append(int(year[x][0]) *1000+int(year[x][1])*100+int(year[x][2])*10+int(year[x][3]))
    for x in range(len(yearlist)):
        file = open(write_path,"a+")
        file.write(s)
        file.write(str(yearlist[x])+ '-12-31')
        file.write(' \000')
        file.close()
        v = net_assets_income_rate(balance_path,profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000')
            file.close()
        else:
            file.write(str(v))
            file.write(' \000')
            file.close()
    
        v = profit_rate(profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
        
        v = main_business_growth_rate(profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
        
        v = net_profit_growth_rate(profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
        
        v = total_assets_growth_rate(balance_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
        
        v = net_assets_growth_rate(balance_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
        
        v = earnings_per_share_growth_rate(profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
        
        v = total_assets_turnover(balance_path,profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1 or v==0:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
        
        v = net_assets_turnover(balance_path,profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1 or v==0:
            file.write('--')
            file.write(' \000')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
        
        v = stock_turnover(balance_path,profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write(' \000')
            file.close()
    
  
        v = accounts_receivable_turnover(balance_path,profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
 
  
        v = return_on_total_assets(balance_path,profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1 or v==0:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
   
        v = net_profit_rate(profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1 or v==0:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
        
        v = gross_profit_rate(profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
        
        
        v = asset_liability_ratio(balance_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write(' \000')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
    
        v = current_ratio(balance_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1 or v==0:
            file.write('--')
            file.write(' \000')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
     
        v = quick_ratio(balance_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
      
    
        v = Long_term_asset_suitability(balance_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1 or v==0:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write(' \000')
            file.close()
   
        v = time_interest_earned_ratio(profit_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1 or v==0:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
   
     
        v = operating_cash_flow_and_liability_ratio(balance_path,cashflow_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v))
            file.write('\000 ')
            file.close()
        
        v = net_profit_cash_amount(profit_path,cashflow_path,yearlist[x])
        file = open(write_path,"a+")
        if  v==-1:
            file.write('--')
            file.write('\000 ')
            file.close()
        else:
            file.write(str(v) + '\n')
            file.close()
            
            
            
            
            
            
            
def main():
    
    path = 'E:\本科毕业论文\TEST\TEST\spider\data'
    text = []
    l = []
    for i in os.walk(path):
        p = i
        text.append(p[0])
    del text[0]
   # print(text[0])
   # print(text[0])
    file = open('E:/本科毕业论文/TEST/TEST/spider/' + 'rate.txt',"w")
    file.close()
    
    flag = 0.0; sum = len( text )
    for x in range(1,len(text)):
        l = (re.findall('\d{2,7}',text[x]))
        path = 'data/' + 'quote' + str(l[0]) +'/'  
        path = 'E:/本科毕业论文/TEST/TEST/spider/' + path
       # calculate(path + 'debt.csv',path + 'lrb.csv',path + 'money.csv')
        write(path + 'debt.csv',path + 'lrb.csv',path + 'money.csv','E:/本科毕业论文/TEST/TEST/spider/rate.txt',str(l[0]) + '\000')
        pos = retain((flag/sum))                                             #完成率
        flag = flag + 1
        print(format(pos,'10.2%'))  
    print('完成') 
    
if __name__ == '__main__':
    main()    
    
    