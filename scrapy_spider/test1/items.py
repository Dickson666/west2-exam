# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ReplyItem(scrapy.Item):
    Type = scrapy.Field()
    Name = scrapy.Field()
    Text = scrapy.Field()
    Time = scrapy.Field()
    Likes = scrapy.Field()
    Rpid = scrapy.Field()
    Replies = scrapy.Field()
    Son = scrapy.Field()

class VideoItem(scrapy.Item):
    Type = scrapy.Field()
    Likes = scrapy.Field()
    Coins = scrapy.Field()
    Stars = scrapy.Field()
    Shares = scrapy.Field()
    Views = scrapy.Field()
    Replies = scrapy.Field()
