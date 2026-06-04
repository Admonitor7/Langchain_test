from common.settings import *
from common.my_mysql_add import *
from DrissionPage import ChromiumPage, ChromiumOptions, WebPage
import time, json
import pyautogui
import shutil
import random


def save_data(key, data):
    """追加数据到 Redis List。"""

    # 如果传入的是复杂数据，先转换成 JSON 字符串。
    if isinstance(data, dict):
        data = json.dumps(data)
    r.rpush(key, data)


def get_main_cookie(id):
    """获取主 Cookie，并处理浏览器隔离。"""

    page = None
    try:
        co = ChromiumOptions()
        co.auto_port()
        co.headless(False)
        co.incognito(True)  # 明确启用无痕模式。

        page = WebPage(chromium_options=co)
        page.listen.start('feed', method='get')

        page.get(f'https://www.toutiao.com/c/user/token/{id}/?tab=all')
        # 如需调试页面加载，可临时增加等待时间。
        # 监听视频请求。
        time.sleep(5)
        # 如需调试页面刷新，可在这里主动刷新页面。
        for req in page.listen.steps(timeout=30):
            res = req.response.body
            if res:
                r.set(id, json.dumps(res))
                logger.info('内容添加成功')
                break
            else:
                print(id)
                page.get(f'https://www.toutiao.com/c/user/token/{id}/?tab=all')
                time.sleep(5)

                page.refresh()
    finally:
        if page:
            page.quit()


while True:
    try:
        sql = '''
            select source_account from t_tt_config order by id desc
        '''
        result = execute_sql_query(sql)
        print(11111, result)

        for i in result['data']:
            get_main_cookie(i['source_account'])

            # 如需降低请求频率，可在这里增加短暂等待。
            # 如需处理最后一条数据，可在这里添加对应逻辑。

        time.sleep(60 * 6)
    except Exception as e:
        logger.info(e)
        time.sleep(30)
