import scrapy
import json
from test1.items import ReplyItem
from test1.items import VideoItem
# from test1.items import VideoItem

class WorkSpider(scrapy.Spider):
    name = "work"

    aid = "" #填入aid

    start_urls = ["https://api.bilibili.com/x/v2/reply/main?next=1&type=1&oid="+aid+"&mode=3"]

    base_url_R = ["https://api.bilibili.com/x/v2/reply/main?next=", "&type=1&oid="+aid+"&mode=3"]

    base_url_SR = ["https://api.bilibili.com/x/v2/reply/reply?oid="+aid+"&pn=","&ps=10&root=","&type=1"]

    base_url_V = "https://api.bilibili.com/x/web-interface/archive/stat?aid=" + aid
    
    page = [1, 1]

    def parse_Video(self, response):

        item = VideoItem()

        text = response.body
        lst = json.loads(text)
        dat = lst['data']

        item['Views'] = dat['view']
        item['Replies'] = dat['reply']
        item['Coins'] = dat['coin']
        item['Shares'] = dat['share']
        item['Likes'] = dat['like']
        item['Stars'] = dat['favorite']
        item['Type'] = 'V'

        yield item

    def parse_SubReply(self, response):
        
        Name = []
        Text = []
        Time = []
        Likes = []

        text = response.body

        lst = json.loads(text)

        rep_lst = lst['data']['replies']

        for i in rep_lst:
            Name.append(i['member']['uname'])
            Text.append(i['content']['message'])
            Likes.append(i['like'])
            Time.append(i['ctime'])
        
        if Name == []:
            # print("Empty!!!")
            return
        
        item = ReplyItem()

        for i in range(len(Name)):
            item['Name'] = Name[i]
            item['Text'] = Text[i]
            item['Likes'] = Likes[i]
            item['Time'] = Time[i]
            item['Rpid'] = response.meta['rid']
            item['Replies'] = '0'
            item['Son'] = 'True'
            item['Type'] = 'R'
            yield item
        
        self.page[1] += 1
        new_url = self.base_url_SR[0] + str(self.page[1]) + self.base_url_SR[1]
        yield scrapy.Request(url=new_url, callback=self.parse_SubReply)

    def parse(self, response):
        
        yield scrapy.Request(url = self.base_url_V, callback=self.parse_Video)

        # return

        Name = []
        Text = []
        Time = []
        Likes = []
        Rid = []
        Replies = []

        # print(type(response))

        # print(response.encoding)

        # try:
        text = response.body
            # with open("test.txt","wb+") as f:
            #     f.write(text)
        # except:
        #     print("QWQ")

        # print("qwq")
        
        lst = json.loads(text)

        # print(type(lst))

        rep_lst = lst['data']['replies']

        for i in rep_lst:
            Rid.append(i['rpid'])
            Name.append(i['member']['uname'])
            Text.append(i['content']['message'])
            Likes.append(i['like'])
            Replies.append(i['count'])
            Time.append(i['ctime'])
        
        if Name == []:
            print("Empty!!!")
            return

        item = ReplyItem()

        for i in range(len(Name)):
            # print("Work", i, "/cy")
            item['Name'] = Name[i]
            item['Text'] = Text[i]
            item['Likes'] = Likes[i]
            item['Replies'] = Replies[i]
            item['Time'] = Time[i]
            item['Type'] = 'R'
            item['Son'] = 'False'
            item['Rpid'] = Rid[i]
            yield item
            # self.root = Rid[i]
            self.page[1] = 1
            new_url = self.base_url_SR[0] + str(self.page[1]) + self.base_url_SR[1] + str(Rid[i]) + self.base_url_SR[2]
            # print(new_url)
            # print("SR")
            yield scrapy.Request(url=new_url, callback=self.parse_SubReply, meta={'rid': Rid[i]})
            # print("FINISH")
            # with open("rpid.txt", "w+") as f:
            #     f.write(Rid[i])
                
            # yield scrapy.Request(callback=Work3Spider.parse)

        # print("????")
        
        # if(response.xpath("/html/body/div[3]/div[2]/div[2]/div[10]/span[2]//text()")[0].extract() == 'true'): # 没有评论了

        #     return
        
        try:
            self.page[0] += 1
            new_url = self.base_url_R[0] + str(self.page[0]) + self.base_url_R[1]
        except Exception as f:
            print(f)
        # print("?!?")

        yield scrapy.Request(url=new_url, callback=self.parse)
