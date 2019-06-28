import jqdatasdk as jq
import os

def write(content, out):
    f = open(out, 'a')
    f.writelines(content)
    f.flush()
    f.close()


def clear(out):
    if os.path.exists(out):
        os.remove(out)


def login():
    jq.auth('18758561268', '561268')


def get_jj_data():
    clear('./bob')
    today = '2019-06-27'
    data = jq.get_price(security='150019.XSHE', start_date='2015-01-01', end_date=today)
    jz = jq.get_extras(info='unit_net_value', security_list='150019.XSHE', start_date='2015-01-01', end_date=today, df=True,count=None)
    result = []
    for indexs in data.index:
        w_time = indexs
        w_close = data.loc[indexs].values[1]
        w_jz = jz.loc[indexs].values[0]
        diff = float(w_close)-float(w_jz)
        diffrea = diff/float(w_close)
        write('{0}--{1}--{2}--{3}--{4}\n'.format(w_time,w_close,w_jz,diff,diffrea),'./bob')    



if __name__ == '__main__':
    login()
    get_jj_data()

    
