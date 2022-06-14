from pprint import pprint
import requests

class RequestHandler:

    def __init__(self, API):
        self.PEXELS_AUTHORIZATION = {"Authorization":API}
        self.request = None
        self.json = None
        self.page = None
        self.total_results = None
        self.photo_results = None
        self.video_results = None
        self.has_next_page = None
        self.has_previous_page = None
        self.next_page = None
        self.prev_page = None


    def pexels_request(self, media, query, page, per_page, orientation='landscape', size='large'):

        urls = {
            'image' : 'https://api.pexels.com/v1/search?',
            'video' : 'https://api.pexels.com/videos/search?'
        }

        url = urls.get(media)

        query = query.replace(" ", "+")
        url = url + "query={}&per_page={}&page={}&orientation={}&size={}".format(query, per_page, page, orientation, size)
        self.__request(url)
        # If there is no json data return None
        return None if not self.request else self.json

    


#""" Private methods """


    def format_request(self, media):
        def ret(dict, key):
            return None if key not in dict.keys() else dict.get(key)

        result = []
        media_result = {}
    
        if media == 'video':
            for video in self.json.get('videos'):
                media_result['Creator'] = ret(video.get('user'), 'name')
                media_result['Media Name'] = ret(video, 'alt')
                media_result['Duration'] = ret(video, 'duration')

                for link in video.get('video_files'):
                    if link.get('quality') == 'hd':
                        media_result['Link'] = ret(link, 'link')
                        break
            
                result.append(media_result)
                media_result = {}
        elif media == 'image':
            for image in self.json.get('photos'):
                media_result['Creator'] = ret(image, 'photographer')
                media_result['Media Name'] = ret(image, 'alt')
                media_result['Link'] = ret(image.get('src'), 'original')

                result.append(media_result)
                media_result = {}
        return result


    def __request(self, url):
        try:
            self.request = requests.get(url, timeout=10, headers=self.PEXELS_AUTHORIZATION)
            self.__update_page_properties()
        except requests.exceptions.RequestException:
            print("Request failed check your internet connection")
            self.request = None
            exit()


    def __update_page_properties(self):
        if self.request.ok:
            self.json = self.request.json()
            try:
                self.page = int(self.json["page"])
            except:
                self.page = None
            try:
                self.total_results = int(self.json["total_results"])
            except:
                self.total_results = None
            try:
                self.photo_results = len(self.json["photos"])
            except:
                self.photo_results = None
            try:
                self.video_results = len(self.json["videos"])
            except:
                self.video_results = None
            try:
                self.next_page = self.json["next_page"]
                self.has_next_page = True
            except:
                self.next_page = None
                self.has_next_page = False
            try:
                self.prev_page = self.json["prev_page"]
                self.has_previous_page = True
            except:
                self.prev_page = None
                self.has_previous_page = False
        else:
            print("Wrong response you might have a wrong API key")
            print(self.request)
            print("API key: {}".format(self.PEXELS_AUTHORIZATION))
            self.request = None
            exit()
