import requests

api_key = 'xxxxxx'  # replace with a valid bingmap api key
base_url = "https://dev.virtualearth.net/REST/v1/Imagery/Map/"
zoom = "20"
maptype = "Aerial"
size = "600,600"


def bing_getaerial(latlon):
    url = base_url + maptype + '/' + latlon + '/' + zoom + '?mapsize=' + size + "&fmt=png&key=" + api_key
    response = requests.get(url)
    with open('./mapsamples/testmap.png', 'wb') as file:   # save the image to .png file
        file.write(response.content)
    return


def extract(list, idx):   # to extract the idxth item in all sublists
    return [item[idx] for item in list]


def bing_avimarker(center_latlon, edge_latlons, avi_all, avi_comb_all, ap_names_all):
    marker_url_all = 'pp=' + center_latlon + ';66'
    headers = {'Content-Length': 'insertLengthOfHTTPBody',
               'Content-Type': 'text/plain; charset=utf-8'}
    num_aps = len(avi_all[0])
    for idx in range(num_aps):
        for edge_latlon, avi in zip(edge_latlons, extract(avi_all, idx)):

            if avi == True:
                marker_url_all = marker_url_all + '&pp=' + edge_latlon + ';79'
            else:
                marker_url_all = marker_url_all + '&pp=' + edge_latlon + ';80'
        url = base_url + maptype + '/' + center_latlon + '/' + zoom + '?mapsize=' + size + "&fmt=png&key=" + api_key

        response = requests.post(url, headers=headers, data=marker_url_all)
        marker_url_all = 'pp=' + center_latlon + ';66'
        # save result image for each AP
        with open('./mapsamples/testmap_avi_' + ap_names_all[0][idx] + '.png', 'wb') as file:  # save the image to .png file
            file.write(response.content)

    for edge_latlon, avi in zip(edge_latlons, avi_comb_all):

        if avi == True:
            marker_url_all = marker_url_all + '&pp=' + edge_latlon + ';79'
        else:
            marker_url_all = marker_url_all + '&pp=' + edge_latlon + ';80'
    url = base_url + maptype + '/' + center_latlon + '/' + zoom + '?mapsize=' + size + "&fmt=png&key=" + api_key

    response = requests.post(url, headers=headers, data=marker_url_all)
    # save result image for all APs together
    with open('./mapsamples/testmap_avi_allAPs.png', 'wb') as file:  # save the image to .png file
        file.write(response.content)
    return
