# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
import pymysql

class VideoPipeline:
    db = None

    cursor = None

    sql = "create table video(\
           Likes char(20),\
           Coins char(20),\
           Stars char(20),\
           Shares char(20),\
           Views char(20),\
           Replies char(20)\
           )"
    
    def open_spider(self, spider):
        print("V Start...")
        self.db = pymysql.connect(host='127.0.0.1', port=3306, user='root', password = '123456', database= 'test')
        self.cursor = self.db.cursor()
        self.cursor.execute("drop table if exists video")
        self.cursor.execute(self.sql)

    def process_item(self, item, spider):
        # print("I'm comming", item['Type'])
        if item['Type'] != 'V':
            # print("?????")
            return item
        try:
            self.cursor.execute("insert into video\
                                (Likes, Coins, Stars, Shares, Views, Replies)\
                                values\
                                ( '%s' , '%s' , '%s' , '%s' , '%s' , '%s' )"\
                                % (item['Likes'], item['Coins'], item['Stars'], item['Shares'], item['Views'], item['Replies']) )
            self.db.commit()
        except Exception as f:
            print(f)
            self.db.rollback()
    def close_spider(self, spider):
        print("V Finished!")
        self.cursor.close()
        self.db.close()
class ReplyPipeline:
    db = None

    cursor = None

    sql = "create table reply(\
           Rpid char(20),\
           Name char(100),\
           Text char(200),\
           Time char(30),\
           Likes char(20),\
           Replies char(20),\
           Son char(10)\
           )"
    
    def open_spider(self, spider):
        print("R Start...")
        self.db = pymysql.connect(host='127.0.0.1', port=3306, user='root', password = '123456', database= 'test')
        self.cursor = self.db.cursor()
        self.cursor.execute("drop table if exists reply")
        self.cursor.execute(self.sql)

    def process_item(self, item, spider):
        # print("true dude")
        # if item['Type'] != 'R':
        #     return item
        try:
            self.cursor.execute("insert into reply\
                                (Rpid, Name, Text, Time, Likes, Replies, Son)\
                                values\
                                ( '%s' , '%s' , '%s' , '%s' , '%s' , '%s' , '%s' )"\
                                % (item['Rpid'], item['Name'], item['Text'], item['Time'], item['Likes'], item['Replies'], item['Son']) )
            self.db.commit()
        except Exception as f:
            print(f)
            self.db.rollback()
    def close_spider(self, spider):
        print("R Finished!")
        self.cursor.close()
        self.db.close()
