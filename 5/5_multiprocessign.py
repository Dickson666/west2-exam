import requests
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing
from lxml import etree

head = {
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36'
}

base_url = "https://jwch.fzu.edu.cn/"

# Name = {}

# works = 0
def work_on_page(url, Name, Time):
    # global works
    # works += 1
    # works += 1
    text = requests.get(url= url, headers= head)
    # print(text.encoding)
    text = etree.HTML(text.content.decode('utf-8'))
    text_list = text.xpath('/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[1]/ul//li')
    multiprocessing.Lock().acquire()
    for i in text_list: 
        owner = str(i.xpath('text()')[1])
        time = str(i.xpath('span/text()')[0])
        Name.append((owner, time))
        # Time.append(time)
        # print(owner, time)
    multiprocessing.Lock().release()

if __name__ == '__main__':

    Times = []

    first_page = requests.get(url= base_url + "jxtz.htm", headers= head)

    open('test.html', 'wb').write(first_page.content)

    first_page = etree.HTML(first_page.text)

    page = first_page.xpath('/html/body/div[1]/div[2]/div[2]/div/div/div[3]/div[2]/div[1]/div/span[1]/span[9]/a/text()')[0]

    page = int(page)

    print(page)

    
    Names = multiprocessing.Manager().list()
    Time = multiprocessing.Manager().list()

    t = multiprocessing.Pool(50)

    t.apply_async(work_on_page, (base_url + "jxtz.htm", Names, Time, ))

    for i in range(1, page):
        t.apply_async(work_on_page, (base_url + "jxtz/" + str(i) + ".htm", Names, Time,))

    t.close()

    t.join()

    # a = "QWQ"
    # a.replace()
    # print(type(a))

    print(len(Names))

    # Names.sort()

    # Time.sort()

    print("!")
    Name = {}

    for k in range(len(Names)):
        i, j = Names[k]
        # j = Time[k]
        if not i in Name:
            Name.update({i: []})
        Name[i].append(j)
    for i in Name:
        # print(i, len(i))
        for j in range(len(Name[i])):
            time = ""
            for l in range(len(Name[i][j])):
                if not (Name[i][j][l] > '9' or Name[i][j][l] < '0'):
                    time += Name[i][j][l]
            # print(time[0])
            # print(type(time))
            # time.replace("\n","")
            # time.replace("-", "")
            # print(time)
            Name[i][j] = time
            if not time in Times:
                Times.append(int(time))

    Times.sort()

    # for i in Times:
    #     print(i, end=" ")

    time_len = len(Times)
    name_len = len(Name)
    X = np.array(Times)
    Y = np.zeros((name_len ,time_len))
    # f = open("tst.txt", "w")
    num = 0
    for i in Name:
        Name[i].sort()
        # f.write("\n")
        k = 0
        for j in range(len(Name[i])):
            # f.write(Name[i][j])
            while Times[k] < int(Name[i][j]):
                k += 1
            if Times[k] == int(Name[i][j]):
                Y[num][k] += 1
        num += 1
    np.set_printoptions(threshold=time_len)
    # print(Y[0])

    plt.xlabel('time')
    plt.ylabel('number')

    num = 0
    for i in Name:
        plt.plot(X, Y[num], label=num)
        num += 1
    # print(works)
    plt.legend()
    plt.show()
# work_on_page(base_url + "jxtz.htm")
